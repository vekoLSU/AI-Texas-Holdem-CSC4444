# Poker bot test server
import asyncio
import websockets
import json
import random

class PokerTestServer:
    def __init__(self):
        self.clients = {}
        self.game_state = {
            'phase': 'preflop',
            'pot': 0,
            'community_cards': [],
            'current_bet': 0
        }

    def deal_cards(self, count=2, strong=False):
        """Deal 'count' cards. If strong=True, deal strong starting hand (e.g., AK, AA, KK)."""
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['h', 'd', 'c', 's']
        if strong:
            # Return AK suited or AA/KK
            strong_hands = [ ['A', 'K'], ['A', 'A'], ['K', 'K'] ]
            hand = random.choice(strong_hands)
            cards = [ (hand[0], random.choice(suits)), (hand[1], random.choice(suits)) ]
        else:
            cards = []
            for _ in range(count):
                rank = random.choice(ranks)
                suit = random.choice(suits)
                cards.append( (rank, suit) )
        return cards
    
    async def handle_client(self, websocket):
        client_id = id(websocket)
        print(f"New client: {client_id}")
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

                # Start game simulation
                await self.simulate_game(websocket, client_id, player_name)

        except websockets.exceptions.ConnectionClosed:
            print(f"❌ Client disconnected: {client_id}")
        except Exception as e:
            print(f"⚠️ Error handling client: {e}")
        finally:
            if client_id in self.clients:
                del self.clients[client_id]
    
    async def simulate_game(self, websocket, client_id, player_name):
        """Simulate a few poker hands"""
        for hand_num in range(1, 1001):
            print(f"\nHand #{hand_num}")
            print(f"\n🃏 Starting hand #{hand_num}")
            # Bot gets strong cards, opponents random
            hole_cards = self.deal_cards(2, strong=True)
            opp1_cards = self.deal_cards(2)
            opp2_cards = self.deal_cards(2)

            # Send game state - preflop
            await websocket.send(json.dumps({
                'type': 'state',
                'hand': hand_num,
                'phase': 'PREFLOP',
                'pot': 30,
                'currentBet': 20,
                'cards': hole_cards,
                'communityCards': [],
                'currentPlayer': player_name,
                'players': [
                    {'id': player_name, 'chips': 980, 'bet': 0, 'folded': False, 'position': 0, 'cards': hole_cards},
                    {'id': 'Opponent1', 'chips': 940, 'bet': 10, 'folded': False, 'position': 1, 'cards': opp1_cards},
                    {'id': 'Opponent2', 'chips': 920, 'bet': 20, 'folded': False, 'position': 2, 'cards': opp2_cards}
                ]
            }))

            # Wait for bot action
            try:
                action_msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                action = json.loads(action_msg)
                print(f"🤖 Bot preflop action: {action}")
                bot_action = action.get('action', 'fold')
                if bot_action == 'fold':
                    print("Bot folded, starting next hand...")
                    continue
            except asyncio.TimeoutError:
                print("⏱️ Bot took too long, folding automatically")
                continue

            # Simulate Opponent1 and Opponent2 actions (randomized, with bluffing and strong hand logic)
            def opponent_action(cards, phase):
                # Simple logic: strong hands raise, weak hands fold/call, random bluff
                ranks = [c[0] for c in cards]
                strong = ('A' in ranks and 'K' in ranks) or ranks.count('A') == 2 or ranks.count('K') == 2
                if strong:
                    return random.choice(['raise', 'call'])
                elif random.random() < 0.15:
                    return 'raise'  # Bluff
                else:
                    return random.choice(['call', 'fold'])

            opp1_action = opponent_action(opp1_cards, 'PREFLOP')
            opp2_action = opponent_action(opp2_cards, 'PREFLOP')
            print(f"Opponent1 preflop action: {opp1_action}")
            print(f"Opponent2 preflop action: {opp2_action}")
            
            # await asyncio.sleep(0.5)  # Removed for speed
            
            # Deal flop
            flop = self.deal_cards(3)
            await websocket.send(json.dumps({
                'type': 'state',
                'hand': hand_num,
                'phase': 'FLOP',
                'pot': 60,
                'currentBet': 0,
                'cards': hole_cards,
                'communityCards': flop,
                'currentPlayer': player_name,
                'players': [
                    {'id': player_name, 'chips': 960, 'bet': 0, 'folded': False, 'position': 0, 'cards': hole_cards},
                    {'id': 'Opponent1', 'chips': 930, 'bet': 0, 'folded': False, 'position': 1, 'cards': opp1_cards},
                    {'id': 'Opponent2', 'chips': 920, 'bet': 0, 'folded': False, 'position': 2, 'cards': opp2_cards}
                ]
            }))
            # Opponent actions on flop
            opp1_action = opponent_action(opp1_cards, 'FLOP')
            opp2_action = opponent_action(opp2_cards, 'FLOP')
            print(f"Opponent1 flop action: {opp1_action}")
            print(f"Opponent2 flop action: {opp2_action}")
            
            try:
                action_msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                action = json.loads(action_msg)
                print(f"🤖 Bot flop action: {action}")
            except asyncio.TimeoutError:
                print("⏱️ Bot took too long")
            
            # await asyncio.sleep(0.5)  # Removed for speed
            
            # Deal turn
            turn = self.deal_cards(1)
            await websocket.send(json.dumps({
                'type': 'state',
                'hand': hand_num,
                'phase': 'TURN',
                'pot': 100,
                'currentBet': 0,
                'cards': hole_cards,
                'communityCards': flop + turn,
                'currentPlayer': player_name,
                'players': [
                    {'id': player_name, 'chips': 940, 'bet': 0, 'folded': False, 'position': 0, 'cards': hole_cards},
                    {'id': 'Opponent1', 'chips': 910, 'bet': 0, 'folded': False, 'position': 1, 'cards': opp1_cards},
                    {'id': 'Opponent2', 'chips': 900, 'bet': 0, 'folded': False, 'position': 2, 'cards': opp2_cards}
                ]
            }))
            # Opponent actions on turn
            opp1_action = opponent_action(opp1_cards, 'TURN')
            opp2_action = opponent_action(opp2_cards, 'TURN')
            print(f"Opponent1 turn action: {opp1_action}")
            print(f"Opponent2 turn action: {opp2_action}")
            
            try:
                action_msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                action = json.loads(action_msg)
                print(f"🤖 Bot turn action: {action}")
            except asyncio.TimeoutError:
                print("⏱️ Bot took too long")
            
            # await asyncio.sleep(0.5)  # Removed for speed
            
            # Deal river
            river = self.deal_cards(1)
            await websocket.send(json.dumps({
                'type': 'state',
                'hand': hand_num,
                'phase': 'RIVER',
                'pot': 150,
                'currentBet': 0,
                'cards': hole_cards,
                'communityCards': flop + turn + river,
                'currentPlayer': player_name,
                'players': [
                    {'id': player_name, 'chips': 915, 'bet': 0, 'folded': False, 'position': 0, 'cards': hole_cards},
                    {'id': 'Opponent1', 'chips': 885, 'bet': 0, 'folded': False, 'position': 1, 'cards': opp1_cards},
                    {'id': 'Opponent2', 'chips': 875, 'bet': 0, 'folded': False, 'position': 2, 'cards': opp2_cards}
                ]
            }))
            # Opponent actions on river
            opp1_action = opponent_action(opp1_cards, 'RIVER')
            opp2_action = opponent_action(opp2_cards, 'RIVER')
            print(f"Opponent1 river action: {opp1_action}")
            print(f"Opponent2 river action: {opp2_action}")
            
            try:
                action_msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                action = json.loads(action_msg)
                print(f"🤖 Bot river action: {action}")
            except asyncio.TimeoutError:
                print("⏱️ Bot took too long")
            
            # await asyncio.sleep(0.5)  # Removed for speed
            
            # Showdown - Bot wins!
            final_board = flop + turn + river
            print(f"🎴 Final board: {final_board}")
            print(f"🃏 Bot cards: {hole_cards}")
            print(f"🏆 {player_name} WINS ${150}!")
            
            await websocket.send(json.dumps({
                'type': 'showdown',
                'winner': player_name,
                'pot': 150,
                'winning_hand': 'Best Hand',
                'board': final_board,
                'player_cards': hole_cards
            }))
            
            print(f"✅ Hand #{hand_num} complete\n")
            # await asyncio.sleep(1)  # Removed for speed
        
        print(f"\n🎉 Test session complete for {player_name}")
    
                # Deal random cards
        """Deal random cards"""
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['h', 'd', 'c', 's']
                # Bot gets strong hands if strong=True
        # Deal strong starting hands for testing
        if strong and count == 2:
            strong_hands = [
                ['As', 'Ah'],  # Pocket Aces
                ['Kh', 'Kd'],  # Pocket Kings
                ['Ac', 'Kd'],  # Ace-King suited
                ['Qh', 'Qd'],  # Pocket Queens
                ['Ad', 'Qc'],  # Ace-Queen
            ]
            return random.choice(strong_hands)
        
        cards = []
        for _ in range(count):
            rank = random.choice(ranks)
            suit = random.choice(suits)
            cards.append(f"{rank}{suit}")
        return cards

async def main():
    server = PokerTestServer()
    
    print("=" * 50)
    print("🎰  Poker Test Server  🎰")
    print("=" * 50)
    print("\nServer starting on ws://localhost:8080")
    print("Waiting for bot connections...\n")
    
    async with websockets.serve(server.handle_client, "localhost", 8080):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
