"""
RL Competition Bot: Uses trained ActorCriticAgent for competition play.

This bot connects the trained PyTorch model to the competition infrastructure via WebSocket.
It handles:
- WebSocket communication with competition server
- State translation (JSON -> 50-dim numpy array)
- Action translation (agent output -> JSON)
- ICM tournament adjustments
- Logging and error handling
- Reconnection and timeout protection
"""

import asyncio
import json
import logging
import time
import traceback
from typing import Dict, Optional
import websockets
import torch
import numpy as np

from poker_bot.training.networks import ActorCriticAgent
from poker_bot.integration.state_translator import StateTranslator
from poker_bot.integration.action_translator import ActionTranslator
from poker_bot.strategy.icm_adjuster import ICMAdjuster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class RLCompetitionBot:
    """Competition bot using trained RL agent."""

    def __init__(
        self,
        agent: ActorCriticAgent,
        player_id: str,
        server_url: str = "ws://localhost:8080/ws",
        api_key: str = "dev",
        table_id: str = "table-1",
        device: str = "cpu",
        use_icm: bool = True,
        deterministic: bool = True
    ):
        """
        Initialize competition bot.

        Args:
            agent: Trained ActorCriticAgent
            player_id: Our player ID
            server_url: WebSocket server URL
            api_key: API key for authentication
            table_id: Table ID to join
            device: PyTorch device ("cpu" or "cuda")
            use_icm: Whether to apply ICM adjustments
            deterministic: Use greedy policy (True) or sample (False)
        """
        self.agent = agent
        self.player_id = player_id
        self.api_key = api_key
        self.table_id = table_id
        self.device = torch.device(device)
        self.deterministic = deterministic

        # Move agent to device and set eval mode
        self.agent = self.agent.to(self.device)
        self.agent.eval()

        # Build WebSocket URL
        self.ws_url = f"{server_url}?apiKey={api_key}&table={table_id}&player={player_id}"

        # Components
        self.state_translator = StateTranslator(starting_chips=1000)
        self.action_translator = ActionTranslator(big_blind=10, min_raise=10)
        self.icm_adjuster = ICMAdjuster(total_players=7, starting_chips=1000) if use_icm else None

        # State tracking
        self.ws = None
        self.hands_played = 0
        self.actions_taken = 0
        self.last_state = None

        # Statistics
        self.stats = {
            "hands": 0,
            "folds": 0,
            "calls": 0,
            "checks": 0,
            "raises": 0,
            "errors": 0
        }

    async def connect_and_play(self):
        """Main bot loop: connect to server and play."""
        logger.info(f"🤖 Connecting to competition server")
        logger.info(f"🎯 URL: {self.ws_url}")
        logger.info(f"👤 Player: {self.player_id}")

        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self.ws = ws
                    logger.info(f"✅ Connected to server")

                    # Send join message
                    await self.send_message({"type": "join"})
                    logger.info(f"📤 Sent JOIN message")

                    # Main game loop
                    while True:
                        try:
                            # Receive message with timeout
                            msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                            await self.handle_message(msg)

                        except asyncio.TimeoutError:
                            logger.warning("⏰ No message received for 30s, sending heartbeat")
                            # Server might be idle, just wait

                        except websockets.exceptions.ConnectionClosed:
                            logger.error("❌ Connection closed by server")
                            break

            except Exception as e:
                retry_count += 1
                logger.error(f"❌ Connection error (attempt {retry_count}/{max_retries}): {e}")

                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.info(f"⏳ Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("❌ Max retries reached, giving up")
                    raise

        logger.info("👋 Bot shutting down")
        self.print_statistics()

    async def send_message(self, message: Dict):
        """Send JSON message to server."""
        if self.ws:
            await self.ws.send(json.dumps(message))
            logger.debug(f"📤 Sent: {message}")

    async def handle_message(self, raw_msg: str):
        """Handle incoming message from server."""
        try:
            msg = json.loads(raw_msg)
            msg_type = msg.get("type")

            logger.debug(f"📨 Received: {msg_type}")

            if msg_type == "state":
                await self.handle_state_update(msg)

            elif msg_type == "error":
                error_msg = msg.get("message", "Unknown error")
                logger.error(f"⚠️ Server error: {error_msg}")
                self.stats["errors"] += 1

            elif msg_type == "showdown":
                self.handle_showdown(msg)

            else:
                logger.debug(f"📨 Unknown message type: {msg_type}")

        except json.JSONDecodeError:
            logger.error(f"⚠️ Invalid JSON: {raw_msg}")

        except Exception as e:
            logger.error(f"❌ Error handling message: {e}")
            logger.error(traceback.format_exc())

    async def handle_state_update(self, json_state: Dict):
        """Handle game state update and make decision if it's our turn."""
        try:
            state = json_state.get("state", {})
            phase = state.get("phase", "UNKNOWN")
            pot = state.get("pot", 0)
            to_act_idx = state.get("toActIdx", -1)
            table = state.get("table", {})
            players = table.get("players", [])

            # Check if it's our turn
            if to_act_idx >= 0 and to_act_idx < len(players):
                current_player_id = players[to_act_idx].get("id") if players[to_act_idx] else None

                if current_player_id == self.player_id:
                    # IT'S OUR TURN!
                    logger.info(f"\n{'='*60}")
                    logger.info(f"🎴 Phase: {phase} | Pot: ${pot}")

                    # Make decision
                    await self.make_decision(json_state)

                    logger.info(f"{'='*60}\n")

            # Track state for next decision
            self.last_state = json_state

        except Exception as e:
            logger.error(f"❌ Error in state update: {e}")
            logger.error(traceback.format_exc())

    async def make_decision(self, json_state: Dict):
        """Make decision using trained RL agent."""
        decision_start = time.time()

        try:
            # 1. Translate state to numpy arrays
            actor_state, critic_state = self.state_translator.translate(json_state, self.player_id)
            game_state_info = self.state_translator.get_state_info(json_state, self.player_id)

            logger.debug(f"Actor state shape: {actor_state.shape}")
            logger.debug(f"Critic state shape: {critic_state.shape}")

            # 2. Convert to tensors
            actor_tensor = torch.FloatTensor(actor_state).unsqueeze(0).to(self.device)
            critic_tensor = torch.FloatTensor(critic_state).unsqueeze(0).to(self.device)

            # 3. Get action from agent
            with torch.no_grad():
                action_idx, amount_ratio, value = self.agent.get_action_and_value(
                    actor_tensor,
                    critic_tensor,
                    deterministic=self.deterministic
                )

            logger.info(f"🤖 Agent decision: action={action_idx}, amount={amount_ratio:.3f}, value={value:.3f}")

            # 4. Translate to competition action
            action_json = self.action_translator.translate(
                action_idx=action_idx,
                amount_ratio=amount_ratio,
                game_state=game_state_info
            )

            action_name = action_json.get("action")
            action_amount = action_json.get("amount", 0)

            # 5. Apply ICM adjustments if enabled
            if self.icm_adjuster:
                # Estimate hand strength (we don't have this from RL agent, use value as proxy)
                hand_strength = (value + 1.0) / 2.0  # Map value to [0, 1]
                hand_strength = np.clip(hand_strength, 0.0, 1.0)

                action_name, action_amount = self.icm_adjuster.adjust_action(
                    action=action_name,
                    amount=action_amount,
                    game_state=game_state_info,
                    hand_strength=hand_strength
                )

                action_json["action"] = action_name
                action_json["amount"] = action_amount

            # 6. Validate action
            if not self.action_translator.validate_action(action_json, game_state_info):
                logger.warning(f"⚠️ Invalid action {action_json}, using fallback")
                action_json = self.action_translator.fallback_action(game_state_info)

            # 7. Send action
            await self.send_message(action_json)

            # Update stats
            self.actions_taken += 1
            action_name = action_json.get("action", "UNKNOWN")
            if action_name in self.stats:
                self.stats[action_name.lower() + "s"] += 1

            decision_time = time.time() - decision_start
            logger.info(f"🎲 Action: {action_json['action']}" +
                       (f" ${action_json['amount']}" if action_json.get('amount', 0) > 0 else ""))
            logger.info(f"⏱️ Decision time: {decision_time:.3f}s")

            # Safety check: warn if close to timeout
            if decision_time > 8.0:
                logger.warning(f"⚠️ Decision took {decision_time:.1f}s (close to 10s timeout!)")

        except Exception as e:
            logger.error(f"❌ Error making decision: {e}")
            logger.error(traceback.format_exc())

            # Fallback action
            try:
                game_state_info = self.state_translator.get_state_info(json_state, self.player_id)
                fallback = self.action_translator.fallback_action(game_state_info)
                await self.send_message(fallback)
                logger.info(f"🆘 Sent fallback action: {fallback}")
                self.stats["errors"] += 1
            except Exception as e2:
                logger.error(f"❌ Fallback also failed: {e2}")

    def handle_showdown(self, msg: Dict):
        """Handle showdown results."""
        winner = msg.get("winner", "Unknown")
        pot = msg.get("pot", 0)
        winning_hand = msg.get("winning_hand", "")

        self.hands_played += 1
        self.stats["hands"] += 1

        logger.info(f"\n{'='*60}")
        logger.info(f"🎰 SHOWDOWN")
        logger.info(f"{'='*60}")

        if winner == self.player_id:
            logger.info(f"🏆 YOU WIN ${pot}!")
            logger.info(f"✨ Winning hand: {winning_hand}")
        else:
            logger.info(f"😞 {winner} wins ${pot}")

        logger.info(f"📊 Hands played: {self.hands_played}")
        logger.info(f"{'='*60}\n")

    def print_statistics(self):
        """Print bot statistics."""
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 BOT STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Hands played: {self.stats['hands']}")
        logger.info(f"Actions taken: {self.actions_taken}")
        logger.info(f"  Folds: {self.stats['folds']}")
        logger.info(f"  Calls: {self.stats['calls']}")
        logger.info(f"  Checks: {self.stats['checks']}")
        logger.info(f"  Raises: {self.stats['raises']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"{'='*60}\n")


async def main():
    """Test entry point."""
    import sys

    # Configuration
    PLAYER_ID = sys.argv[1] if len(sys.argv) > 1 else "rl_bot_test"
    MODEL_PATH = sys.argv[2] if len(sys.argv) > 2 else "trained_models/agent_final.pt"
    SERVER_URL = sys.argv[3] if len(sys.argv) > 3 else "ws://localhost:8080/ws"

    logger.info(f"""
╔══════════════════════════════════════════════════════════╗
║        🃏  RL COMPETITION BOT  🃏                       ║
║           Trained Actor-Critic Agent                     ║
╚══════════════════════════════════════════════════════════╝
    """)

    # Load trained agent
    logger.info(f"📂 Loading model from: {MODEL_PATH}")
    agent = ActorCriticAgent(actor_state_dim=50, critic_state_dim=70)
    agent.load(MODEL_PATH)
    logger.info(f"✅ Model loaded successfully")

    # Create bot
    bot = RLCompetitionBot(
        agent=agent,
        player_id=PLAYER_ID,
        server_url=SERVER_URL,
        api_key="dev",
        table_id="table-1",
        device="cpu",
        use_icm=True,
        deterministic=True
    )

    try:
        await bot.connect_and_play()
    except KeyboardInterrupt:
        logger.info("\n👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    asyncio.run(main())
