# Poker bot test server - replays historical hand histories
import asyncio
import websockets
import json
import os
from typing import List, Dict, Optional


class PokerTestServer:
    def __init__(self, log_file: Optional[str] = None):
        self.clients = {}

        # Determine which log file to use
        if log_file:
            self.log_file = log_file
        else:
            # Check environment variable
            env_log = os.environ.get('TEST_SERVER_LOG')
            if env_log:
                self.log_file = env_log
            else:
                # Default to first JSONL file in historical_logs/
                import glob
                jsonl_files = glob.glob('historical_logs/*.jsonl')
                if jsonl_files:
                    self.log_file = sorted(jsonl_files)[0]
                else:
                    raise FileNotFoundError(
                        "No JSONL log files found. Please set TEST_SERVER_LOG "
                        "or provide log file path."
                    )

        print(f"📂 Loading hands from: {self.log_file}")
        self.hands = self.load_hands()
        print(f"✅ Loaded {len(self.hands)} hands")

    def load_hands(self) -> List[Dict]:
        """Load hands from JSONL file"""
        hands = []
        with open(self.log_file, 'r') as f:
            for line in f:
                if line.strip():
                    hands.append(json.loads(line))
        return hands

    async def handle_client(self, websocket):
        client_id = id(websocket)
        print(f"✅ New client connected: {client_id}")

        try:
            # Wait for join message
            join_msg = await websocket.recv()
            join_data = json.loads(join_msg)

            if join_data.get('type') == 'join':
                # Extract player name from connection
                player_name = 'TestBot'
                self.clients[client_id] = {
                    'websocket': websocket,
                    'name': player_name,
                    'chips': 1000
                }

                print(f"🎮 Player joined: {player_name}")

                # Start game replay
                await self.replay_hands(websocket, client_id, player_name)

        except websockets.exceptions.ConnectionClosed:
            print(f"❌ Client disconnected: {client_id}")
        except Exception as e:
            print(f"⚠️ Error handling client: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if client_id in self.clients:
                del self.clients[client_id]

    async def replay_hands(self, websocket, client_id, player_name):
        """Replay hands from JSONL log file"""
        for hand_num, hand_data in enumerate(self.hands, start=1):
            print(f"\n🃏 Starting hand #{hand_num} (ID: {hand_data['hand_id']})")

            hero_cards = hand_data['hero_cards']
            phases = hand_data['phases']

            # Replay each phase
            for phase_name in ['PREFLOP', 'FLOP', 'TURN', 'RIVER']:
                if phase_name not in phases:
                    continue

                phase = phases[phase_name]

                # Build players list for this phase, replacing Hero with TestBot
                players_list = []
                for player in phase['players']:
                    player_copy = player.copy()
                    if player['id'] == hand_data['hero_name']:
                        player_copy['id'] = player_name
                        player_copy['cards'] = hero_cards
                    else:
                        # Hide opponent cards
                        player_copy['cards'] = None
                    players_list.append(player_copy)

                # Send state message
                state_msg = {
                    'type': 'state',
                    'hand': hand_num,
                    'phase': phase_name,
                    'pot': phase['pot'],
                    'currentBet': phase['currentBet'],
                    'cards': hero_cards,
                    'communityCards': phase['communityCards'],
                    'currentPlayer': player_name,
                    'players': players_list
                }

                print(f"  📤 Sending {phase_name} state (pot: ${phase['pot']}, "
                      f"community: {phase['communityCards']})")
                await websocket.send(json.dumps(state_msg))

                # Wait for bot action with timeout
                try:
                    action_msg = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=10.0
                    )
                    action = json.loads(action_msg)
                    action_type = action.get('action', 'unknown')
                    amount = action.get('amount', 0)

                    if amount:
                        print(f"  🤖 Bot action: {action_type} ${amount}")
                    else:
                        print(f"  🤖 Bot action: {action_type}")

                    # Check if bot folded
                    if action_type == 'fold':
                        print(f"  ⚠️ Bot folded, ending hand early")
                        break

                except asyncio.TimeoutError:
                    print(f"  ⏱️ Bot took too long, folding automatically")
                    break

            # Send showdown message
            showdown = hand_data['showdown']

            # Replace Hero with TestBot in winner name
            winner = showdown['winner']
            if winner == hand_data['hero_name']:
                winner = player_name

            showdown_msg = {
                'type': 'showdown',
                'winner': winner,
                'pot': showdown['pot'],
                'winning_hand': 'Historical Result',
                'board': showdown['final_board'],
                'player_cards': hero_cards
            }

            await websocket.send(json.dumps(showdown_msg))

            if winner == player_name:
                print(f"  🏆 {player_name} WINS ${showdown['pot']}!")
            else:
                print(f"  💔 {winner} wins ${showdown['pot']}")

            print(f"✅ Hand #{hand_num} complete")

        print(f"\n🎉 Test session complete - replayed {len(self.hands)} hands")


async def main():
    # Check for custom log file argument
    import sys
    log_file = sys.argv[1] if len(sys.argv) > 1 else None

    server = PokerTestServer(log_file=log_file)

    print("=" * 60)
    print("🎰  Poker Test Server - Historical Hand Replay  🎰")
    print("=" * 60)
    print(f"\nReplaying {len(server.hands)} hands from real poker history")
    print("Server starting on ws://localhost:8080")
    print("Waiting for bot connections...\n")

    async with websockets.serve(server.handle_client, "localhost", 8080):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo use this server:")
        print("1. Run: python3 tools/convert_hh_to_jsonl.py <input.txt> <output.jsonl>")
        print("2. Set TEST_SERVER_LOG env var, or place JSONL in historical_logs/")
        print("3. Run: python3 test_server.py")
