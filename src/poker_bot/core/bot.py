import asyncio
import json
import os
import sys
import threading
import time
from typing import Dict, List, Optional
from websocket import WebSocketApp, WebSocketConnectionClosedException

#WS_URL_TEMPLATE = "ws://localhost:8080/ws?apiKey={apiKey}&table={table}&player={player}"
WS_URL_TEMPLATE = "wss://texasholdem-871757115753.northamerica-northeast1.run.app/ws?apiKey={apiKey}&table={table}&player={player}"

class PlayerClient:
    def __init__(self, player_id: str, api_key: str, table_id: str):
        self.player_id = player_id
        self.api_key = api_key
        self.table_id = table_id
        self.ws: WebSocketApp | None = None
        self.my_seat = None
        self.to_act_idx = -1
        self.phase = "WAITING"
        self.lock = threading.Lock()
        
        # Track betting state across phases
        self.last_pot = 0
        self.last_phase = "WAITING"
        self.pot_at_phase_start = 0
        self.my_bet_this_phase = 0
        self.actions_this_phase = []

    def on_open(self, ws):
        print(f"[{self.player_id}] Connected")
        ws.send(json.dumps({"type": "join"}))
        print(f"[{self.player_id}] Waiting for game state...")

    def on_error(self, ws, error):
        print(f"[{self.player_id}] error:", error)

    def on_close(self, ws, status_code, msg):
        print(f"[{self.player_id}] connection closed:", status_code, msg)

    def on_message(self, ws, message: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            print(f"[{self.player_id}] non-JSON:", message)
            return

        if data.get("type") != "state":
            print(f"[{self.player_id}] msg:", data)
            return

        state = data["state"]
        table = state.get("table", {})
        
        # Show connected players on first state received
        if not hasattr(self, 'first_state_received'):
            self.first_state_received = True
            players = table.get("players", [])
            
            print(f"\n{'='*60}")
            print(f"  🎮 TABLE JOINED - {table.get('id', 'unknown')}")
            print(f"{'='*60}")
            print(f"\n  👥 CONNECTED PLAYERS:")
            
            for i, player in enumerate(players):
                player_id = player.get("id", "?")
                chips = player.get("chips", 0)
                is_me = " ⭐ (YOU)" if player_id == self.player_id else ""
                print(f"    [{i}] {player_id}: ${chips}{is_me}")
            
            print(f"\n  💰 Current Pot: ${state.get('pot', 0)}")
            print(f"  🎲 Phase: {state.get('phase', 'WAITING')}")
            print(f"{'='*60}\n")
        
        # Correctly update toActIdx from the game state
        self.to_act_idx = state.get("toActIdx", -1)
        current_phase = state.get("phase", "WAITING")
        current_pot = state.get("pot", 0)
        current_hand = state.get("hand", 0)
        
        # Show player status at the start of each hand (PREFLOP phase)
        if current_phase == "PREFLOP" and (not hasattr(self, 'last_hand_shown') or self.last_hand_shown != current_hand):
            self.last_hand_shown = current_hand
            players = table.get("players", [])
            
            print(f"\n{'='*60}")
            print(f"  🎰 HAND #{current_hand} - STARTING")
            print(f"{'='*60}")
            print(f"\n  👥 CONNECTED PLAYERS:")
            
            for i, player in enumerate(players):
                player_id = player.get("id", "?")
                chips = player.get("chips", 0)
                is_me = " ⭐ (YOU)" if player_id == self.player_id else ""
                print(f"    [{i}] {player_id}: ${chips}{is_me}")
            
            print(f"\n  💰 Starting Pot: ${current_pot}")
            print(f"{'='*60}\n")
        
        # Detect phase change - reset betting tracking
        if current_phase != self.last_phase:
            self.pot_at_phase_start = current_pot
            self.my_bet_this_phase = 0
            self.actions_this_phase = []
            self.last_phase = current_phase
        
        self.phase = current_phase
        players = table.get("players", [])

        for i, player in enumerate(players):
            if player.get("id") == self.player_id:
                self.my_seat = i
                break

        # Only act when it's our turn
        if self.my_seat is not None and self.my_seat == self.to_act_idx and self.phase not in ("WAITING", "SHOWDOWN"):
            print(f"\n{'='*60}")
            print(f"[{self.player_id}] 🎯 OUR TURN TO ACT!")
            print(f"[{self.player_id}] Phase: {self.phase}, Pot: {state.get('pot', 0)}, Hand #{state.get('hand', 0)}")
            print(f"{'='*60}\n")
            
            # Display game situation clearly
            print(f"\n{'🃏'*30}")
            print(f"  GAME SITUATION - {self.phase}")
            print(f"{'🃏'*30}")
            
            # Show MY HAND
            my_cards = table.get("players", [])[self.my_seat].get("cards", [])
            hand_display = " ".join([f"{c.get('rank', '?')}{c.get('suit', '?')}" for c in my_cards])
            print(f"\n  🎴 YOUR HAND: {hand_display}")
            
            # Show BOARD
            board_cards = state.get("board", [])
            if board_cards:
                board_display = " ".join([f"{c.get('rank', '?')}{c.get('suit', '?')}" for c in board_cards])
                print(f"  🎲 BOARD: {board_display}")
            else:
                print(f"  🎲 BOARD: (no community cards yet)")
            
            # Show POT
            print(f"\n  💰 POT: ${state.get('pot', 0)}")
            
            # Show ALL PLAYERS with chips
            print(f"\n  👥 PLAYERS:")
            for i, player in enumerate(players):
                player_id = player.get("id", "?")
                chips = player.get("chips", 0)
                is_me = " ⭐ (YOU)" if i == self.my_seat else ""
                is_acting = " 🎯 (acting)" if i == self.to_act_idx else ""
                chip_status = "ALL-IN" if chips == 0 else f"${chips}"
                print(f"    [{i}] {player_id}: {chip_status}{is_me}{is_acting}")
            
            print(f"\n{'='*60}\n")
            
            # Use the ensemble system to make decisions
            from poker_bot.core.meta_controller import MetaController
            from poker_bot.evaluation.opponent_tracker import OpponentTracker
            from poker_bot.evaluation.hand_evaluator import HandEvaluator

            # Initialize the MetaController with an OpponentTracker
            opponent_tracker = OpponentTracker()
            meta_controller = MetaController(opponent_tracker)
            hand_evaluator = HandEvaluator()

            # Extract hand cards
            hand_cards = [f"{c.get('rank', '')}{c.get('suit', '')[0] if c.get('suit') else ''}" for c in my_cards]
            community_cards = [f"{c.get('rank', '')}{c.get('suit', '')[0] if c.get('suit') else ''}" for c in state.get("board", [])]

            # Get our chip count
            our_chips = table.get("players", [])[self.my_seat].get("chips", 0)
            
            # CRITICAL: If we have 0 chips, we're already all-in!
            if our_chips == 0:
                print(f"[{self.player_id}] ⚠️ ALREADY ALL-IN (chips=0) - can only check/wait")
                # When all-in, we just check (send CALL with 0)
                self.send_action("CALL", 0)
                return

            # Calculate to_call amount from pot changes
            # Since the server doesn't send bet amounts, we infer from pot growth
            
            # Track pot changes to estimate betting
            if not hasattr(self, 'last_pot'):
                self.last_pot = 0
            if not hasattr(self, 'last_phase'):
                self.last_phase = "WAITING"
            if not hasattr(self, 'pot_at_phase_start'):
                self.pot_at_phase_start = 0
            if not hasattr(self, 'my_total_invested'):
                self.my_total_invested = 0
            
            current_pot = state.get('pot', 0)
            
            # Detect phase change - reset betting tracking
            if self.phase != self.last_phase:
                print(f"[{self.player_id}] 🔄 Phase changed: {self.last_phase} → {self.phase}")
                self.pot_at_phase_start = current_pot
                self.last_phase = self.phase
            
            # Calculate pot growth this phase
            pot_growth = current_pot - self.pot_at_phase_start
            
            # Estimate current bet from pot growth
            # This is imperfect but better than nothing
            num_active_players = sum(1 for p in players if p.get("chips", 0) > 0 or p.get("id") == self.player_id)
            
            # If pot grew, someone bet
            estimated_current_bet = 0
            if pot_growth > 0 and num_active_players > 0:
                # Conservative estimate: pot growth divided by number of players who could have bet
                estimated_current_bet = pot_growth // max(num_active_players, 1)
            
            # Try to find explicit currentBet (may not exist)
            explicit_current_bet = (
                data.get("currentBet") or
                state.get("currentBet") or
                table.get("currentBet") or
                0
            )
            
            # Use explicit if available, otherwise use estimate
            current_bet = explicit_current_bet if explicit_current_bet > 0 else estimated_current_bet
            
            # Calculate to_call
            # We don't know exactly what we've bet, so assume we need to match current_bet
            to_call = min(current_bet, our_chips)  # Can't call more than we have
            
            print(f"[{self.player_id}] 💰 BETTING CALCULATION:")
            print(f"   Our chips: {our_chips}")
            print(f"   Pot at phase start: {self.pot_at_phase_start}")
            print(f"   Current pot: {current_pot}")
            print(f"   Pot growth this phase: {pot_growth}")
            print(f"   Active players: {num_active_players}")
            print(f"   Estimated current bet: {estimated_current_bet}")
            print(f"   Amount to call: {to_call}")

            # Calculate hand strength
            hand_strength = hand_evaluator.evaluate_hand_strength(hand_cards, community_cards, self.phase)
            
            print(f"[{self.player_id}] Hand: {hand_cards}, Strength: {hand_strength.get('strength', 0.0)}")

            # Get decision from ensemble
            decision = meta_controller.decide(
                hand_cards=hand_cards,
                community_cards=community_cards,
                hand_strength=hand_strength,  # Now passing actual hand strength
                phase=self.phase,
                pot=state.get('pot', 0),
                to_call=to_call,  # Correctly calculated to_call amount
                our_chips=our_chips,
                position="",  # Placeholder for position
                num_players=len(table.get("players", [])),
                opponent_profiles=[],  # Placeholder for opponent profiles
                current_bet=current_bet,  # Current highest bet
                our_player_id=self.player_id
            )

            # Execute the decision
            action = decision.get("action", "call").lower()  # Keep lowercase for consistency
            amount = decision.get("amount", 0)
            
            print(f"[{self.player_id}] 🎲 Raw decision from agent: action={action}, amount={amount}")
            print(f"[{self.player_id}] 📊 Decision context: to_call={to_call}, hand_strength={hand_strength.get('strength', 0.0)}")
            
            # Handle different actions
            if action == "fold":
                # FOLD - no amount needed
                print(f"[{self.player_id}] 🚫 FOLDING (hand too weak or bet too high)")
                final_action = "FOLD"
                final_amount = 0
            elif action == "call":
                # CALL - send the exact to_call amount
                if to_call == 0:
                    print(f"[{self.player_id}] ✓ CHECKING (no bet to call)")
                else:
                    print(f"[{self.player_id}] 📞 CALLING {to_call}")
                final_action = "CALL"
                final_amount = to_call
            elif action == "raise":
                # RAISE - use the amount from decision
                print(f"[{self.player_id}] 📈 RAISING to {amount}")
                final_action = "RAISE"
                final_amount = int(amount)
            else:
                # Unknown action - default to fold for safety
                print(f"[{self.player_id}] ⚠️ Unknown action '{action}', defaulting to FOLD")
                final_action = "FOLD"
                final_amount = 0
            
            print(f"[{self.player_id}] 📤 Final decision: {final_action}, amount: {final_amount}")
            print(f"[{self.player_id}] Debug - Full decision: {decision}")
            print(f"[{self.player_id}] Debug - Hand: {hand_cards}, Board: {community_cards}")
            print(f"[{self.player_id}] Debug - to_call: {to_call}, current_bet: {current_bet}, our_bet: {self.my_bet_this_phase}")
            
            # Track our action for next time
            if final_action == "CALL":
                self.my_bet_this_phase += final_amount
            elif final_action == "RAISE":
                self.my_bet_this_phase += final_amount
            
            self.actions_this_phase.append(final_action)
            self.last_pot = current_pot
            
            self.send_action(final_action, final_amount)

    def send_action(self, action: str, amount: int = 0):
        if not self.ws or not self.ws.sock or not self.ws.sock.connected:
            print(f"[{self.player_id}] cannot send, not connected")
            return
        if action == "RAISE":
            amount = max(amount,10)
            
        msg = {
            "type": "act",
            "action": action,
            "amount": amount,
        }
        print(f"[{self.player_id}] 📤 Sending to server: {msg}")
        try:
            self.ws.send(json.dumps(msg))
            print(f"[{self.player_id}] ✅ Action sent successfully")
        except WebSocketConnectionClosedException as e:
            print(f"[{self.player_id}] ❌ Send failed:", e)
        except Exception as e:
            print(f"[{self.player_id}] ❌ Unexpected error:", e)

    def run_ws(self):
        url = WS_URL_TEMPLATE.format(apiKey=self.api_key, table=self.table_id, player=self.player_id)
        print(f"[{self.player_id}] Connecting to {url}")

        self.ws = WebSocketApp(
            url,
            on_open=self.on_open,
            on_close=self.on_close,
            on_error=self.on_error,
            on_message=self.on_message,
        )
        self.ws.run_forever()

    def prepare_model_input(self, state: dict) -> List[float]:
        """Prepare the input data for the model based on the game state."""
        # Example: Extract relevant features from the state
        table = state.get("table", {})
        players = table.get("players", [])
        pot = state.get("pot", 0)
        board = state.get("board", [])

        # Convert card ranks to numeric values
        rank_to_numeric = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10,
            'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        input_data = [
            float(pot),
            float(len(board)),
            *[float(player.get("chips", 0)) for player in players],
            *[float(rank_to_numeric.get(card.get("rank", "0"), 0)) for card in board],
            *[0.0] * (50 - len(players) - len(board) - 2)  # Pad with zeros to reach 50 features
        ]
        return input_data

    def prepare_critic_input(self, state: dict) -> List[float]:
        """Prepare the input data for the critic model based on the game state."""
        # Example: Extract relevant features for the critic
        table = state.get("table", {})
        players = table.get("players", [])
        pot = state.get("pot", 0)
        board = state.get("board", [])

        # Convert card ranks to numeric values
        rank_to_numeric = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10,
            'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        critic_input_data = [
            float(pot),
            float(len(board)),
            *[float(player.get("chips", 0)) for player in players],
            *[float(rank_to_numeric.get(card.get("rank", "0"), 0)) for card in board],
            *[0.0] * (70 - len(players) - len(board) - 2)  # Pad with zeros to reach 70 features
        ]
        return critic_input_data

class PokerBot(PlayerClient):
    """
    PokerBot class extending PlayerClient to encapsulate bot-specific logic.
    """
    def __init__(self, player_id: str, api_key: str, table_id: str):
        super().__init__(player_id, api_key, table_id)

    def make_decision(self, state: dict):
        """Override this method to implement bot-specific decision-making."""
        print(f"[{self.player_id}] Making a decision using PokerBot logic...")
        # Example: Always call
        self.send_action("CALL")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python3 player_client.py <playerId> <apiKey> <tableId>")
        sys.exit(1)

    player_id = sys.argv[1]
    api_key = sys.argv[2]
    table_id = sys.argv[3]

    client = PlayerClient(player_id, api_key, table_id)
    threading.Thread(target=client.run_ws, daemon=True).start()

    while True:
        time.sleep(1)
