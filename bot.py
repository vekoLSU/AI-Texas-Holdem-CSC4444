import asyncio
import json
import websockets
import sys
from typing import Optional, Dict, List
from hand_evaluator import HandEvaluator
from decision_engine import DecisionEngine
from opponent_tracker import OpponentTracker

class PokerBot:
    # Poker bot main class
    
    def __init__(self, api_key: str, table: str, player: str, server_url: str = "ws://localhost:8080"):
        self.api_key = api_key
        self.table = table
        self.player = player
        self.server_url = server_url
        
        # Game state
        self.game_state: Optional[Dict] = None
        self.hand_cards: List[str] = []
        self.community_cards: List[str] = []
        
        # Strategy components
        self.hand_evaluator = HandEvaluator()
        self.decision_engine = DecisionEngine()
        self.opponent_tracker = OpponentTracker()
        
        # Stats tracking
        self.hands_played = 0
        self.hands_won = 0
        
    def get_ws_url(self) -> str:
        # Build WebSocket URL
        return f"{self.server_url}/ws?apiKey={self.api_key}&table={self.table}&player={self.player}"
    
    async def connect_and_play(self):
        # Main bot loop
        url = self.get_ws_url()
        print(f"🤖 Connecting to {url}")
        print(f"🎯 Player: {self.player} | Table: {self.table}")
        
        try:
            async with websockets.connect(url) as ws:
                # Join the table
                await self.send_message(ws, {"type": "join"})
                print(f"✅ Joined table as {self.player}")
                
                # Main game loop
                while True:
                    try:
                        msg = await ws.recv()
                        await self.handle_message(ws, msg)
                    except websockets.exceptions.ConnectionClosed:
                        print("❌ Connection closed")
                        break
                        
        except Exception as e:
            print(f"❌ Connection error: {e}")
            raise
    
    async def send_message(self, ws, message: dict):
        # Send message to server
        await ws.send(json.dumps(message))
    
    async def handle_message(self, ws, raw_msg: str):
        # Handle server message
        try:
            msg = json.loads(raw_msg)
            msg_type = msg.get("type")
            
            if msg_type == "state":
                await self.handle_state_update(ws, msg)
            elif msg_type == "showdown":
                self.handle_showdown(msg)
            elif msg_type == "error":
                print(f"⚠️  Server error: {msg.get('message')}")
            else:
                # Log other message types for debugging
                print(f"📨 Received: {msg_type}")
                
        except json.JSONDecodeError:
            print(f"⚠️  Invalid JSON: {raw_msg}")
    
    async def handle_state_update(self, ws, state: dict):
        # Handle game state and act
        import traceback
        self.game_state = state
        try:
            # Extract game info
            phase = state.get("phase", "UNKNOWN")
            pot = state.get("pot", 0)
            hand_num = state.get("hand", 0)
            print(f"[DEBUG] handle_state_update: phase={phase}, pot={pot}, hand_num={hand_num}")

            # Update our cards if provided
            if "cards" in state:
                self.hand_cards = state["cards"]
            if "communityCards" in state:
                self.community_cards = state["communityCards"]

            # Find our player data
            players = state.get("players", [])
            print(f"[DEBUG] handle_state_update: players={players}")
            our_player = self.find_our_player(players)
            print(f"[DEBUG] handle_state_update: our_player={our_player}")

            if not our_player:
                print("[DEBUG] No our_player found, skipping turn.")
                return

            # Check if it's our turn
            current_player = state.get("currentPlayer")
            print(f"[DEBUG] handle_state_update: current_player={current_player}")
            if current_player != self.player:
                # Not our turn, but update opponent tracking
                self.opponent_tracker.observe_state(state, players)
                print("[DEBUG] Not our turn, returning.")
                return

            # IT'S OUR TURN - Make a decision!
            print(f"\n{'='*60}")
            print(f"🎴 Hand #{hand_num} | Phase: {phase} | Pot: ${pot}")
            print(f"🃏 Our cards: {self.hand_cards}")
            print(f"🎯 Community: {self.community_cards}")

            # Make the best decision
            action = await self.make_decision(state, our_player, players)
            print(f"[DEBUG] make_decision returned: {action}")

            if action:
                await self.send_message(ws, action)
                print(f"🎲 Action: {action}")
                print(f"{'='*60}\n")
        except Exception as e:
            print(f"[ERROR] Exception in handle_state_update: {e}")
            traceback.print_exc()
    
    def find_our_player(self, players: List[dict]) -> Optional[dict]:
        # Find our player
        for player in players:
            if player.get("id") == self.player:
                # Defensive: ignore extra fields like 'cards' for opponents
                # Only use our own 'cards' field for decision making
                return {
                    **player,
                    "cards": player.get("cards", self.hand_cards)
                }
        return None
    
    async def make_decision(self, state: dict, our_player: dict, all_players: List[dict]) -> Optional[dict]:
        # Decide action
        
        # Extract key information
        phase = state.get("phase", "UNKNOWN")
        pot = state.get("pot", 0)
        current_bet = state.get("currentBet", 0)
        our_chips = our_player.get("chips", 0)
        our_bet = our_player.get("bet", 0)
        to_call = current_bet - our_bet
        
        import traceback
        try:
            print(f"[DEBUG] make_decision: our_player={our_player}")
            # Position analysis
            position = self.get_position(all_players, our_player)
            num_active_players = sum(1 for p in all_players if not p.get("folded", False))

            # Hand strength evaluation
            hand_strength = self.hand_evaluator.evaluate_hand_strength(
                our_player.get("cards", self.hand_cards),
                self.community_cards,
                phase
            )
            print(f"[DEBUG] make_decision: hand_strength={hand_strength}")

            # Get opponent tendencies
            # Remove 'cards' field for opponents to avoid confusion
            clean_players = [
                {k: v for k, v in p.items() if k != "cards" or p.get("id") == self.player}
                for p in all_players
            ]
            print(f"[DEBUG] make_decision: clean_players={clean_players}")
            opponent_profiles = self.opponent_tracker.get_opponent_profiles(clean_players)
            print(f"[DEBUG] make_decision: opponent_profiles={opponent_profiles}")

            # Use decision engine to determine optimal action
            decision = self.decision_engine.decide(
                hand_cards=our_player.get("cards", self.hand_cards),
                community_cards=self.community_cards,
                hand_strength=hand_strength,
                phase=phase,
                pot=pot,
                to_call=to_call,
                our_chips=our_chips,
                position=position,
                num_players=num_active_players,
                opponent_profiles=opponent_profiles,
                current_bet=current_bet
            )
            print(f"[DEBUG] make_decision: decision={decision}")

            # Format action for server
            action_type = decision["action"]

            if action_type == "fold":
                return {"type": "action", "action": "fold"}
            elif action_type == "call":
                return {"type": "action", "action": "call"}
            elif action_type == "check":
                return {"type": "action", "action": "check"}
            elif action_type == "raise":
                amount = decision.get("amount", current_bet * 2)
                # Ensure we don't raise more than we have
                amount = min(amount, our_chips)
                return {"type": "action", "action": "raise", "amount": int(amount)}
            elif action_type == "all-in":
                return {"type": "action", "action": "raise", "amount": our_chips}

            # Default to check/call
            if to_call == 0:
                return {"type": "action", "action": "check"}
            else:
                return {"type": "action", "action": "call"}
        except Exception as e:
            print(f"[ERROR] Exception in make_decision: {e}")
            traceback.print_exc()
            return None
    
    def get_position(self, players: List[dict], our_player: dict) -> str:
        # Get table position
        active_players = [p for p in players if not p.get("folded", False)]
        num_players = len(active_players)
        
        if num_players <= 2:
            return "heads-up"
        
        # Try to determine position based on player order
        try:
            our_index = next(i for i, p in enumerate(active_players) if p.get("id") == self.player)
            
            if our_index == 0:
                return "early"
            elif our_index < num_players // 2:
                return "middle"
            else:
                return "late"
        except:
            return "unknown"
    
    def handle_showdown(self, msg: dict):
        # Showdown results
        winner = msg.get('winner', 'Unknown')
        pot = msg.get('pot', 0)
        winning_hand = msg.get('winning_hand', '')
        
        print(f"\n{'='*60}")
        print(f"🎰 SHOWDOWN RESULTS")
        print(f"{'='*60}")
        
        if winner == self.player:
            self.hands_won += 1
            print(f"🏆 YOU WIN ${pot}!")
            print(f"✨ Winning hand: {winning_hand}")
            print(f"🎉 Congratulations!")
        else:
            print(f"😞 {winner} wins ${pot}")
            print(f"💔 Better luck next time!")
        
        print(f"\n📊 Session Stats:")
        print(f"   Hands Played: {self.hands_played + 1}")
        print(f"   Hands Won: {self.hands_won}")
        if self.hands_played > 0:
            win_rate = (self.hands_won / (self.hands_played + 1)) * 100
            print(f"   Win Rate: {win_rate:.1f}%")
        print(f"{'='*60}\n")
        
        self.hands_played += 1


async def main():
    # Bot entry point
    
    # Configuration (can be overridden by command line args)
    API_KEY = sys.argv[1] if len(sys.argv) > 1 else "dev"
    TABLE = sys.argv[2] if len(sys.argv) > 2 else "table-1"
    PLAYER = sys.argv[3] if len(sys.argv) > 3 else "bot1"
    SERVER_URL = sys.argv[4] if len(sys.argv) > 4 else "ws://localhost:8080"
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║           🃏  TEXAS HOLD'EM POKER BOT  🃏               ║
║                  Advanced AI Player                      ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    bot = PokerBot(API_KEY, TABLE, PLAYER, SERVER_URL)
    
    try:
        await bot.connect_and_play()
    except KeyboardInterrupt:
        print("\n👋 Bot shutting down...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
