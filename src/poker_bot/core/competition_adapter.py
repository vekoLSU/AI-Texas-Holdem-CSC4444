"""
Adapter to connect the bot to the dtaing11/Texas-HoldEm-Infrastructure
Translates between the engine's protocol and the bot's expected format
"""
import asyncio
import json
import websockets
from poker_bot.core.bot import PlayerClient

class CompetitionAdapter:
    """Adapts bot to work with the competition engine"""
    
    def __init__(self, api_key: str, table: str, player: str, server_url: str):
        self.api_key = api_key
        self.table = table
        self.player = player
        self.server_url = server_url
        self.bot = PokerBot(api_key, table, player, server_url)
        
    def get_ws_url(self) -> str:
        """Construct WebSocket URL for competition engine"""
        return f"{self.server_url}/ws?apiKey={self.api_key}&table={self.table}&player={self.player}"
    
    def translate_state_to_bot(self, engine_state: dict) -> dict:
        """
        Translate competition engine state to bot's expected format
        
        Engine format:
        {
          "type": "state",
          "state": {
            "table": {
              "id": "table-1",
              "players": [...],
              "phase": "PREFLOP",
              "cardOpen": []
            },
            "pot": 30,
            "phase": "PREFLOP",
            "board": [],
            "toActIdx": 2
          }
        }
        
        Bot expects:
        {
          "type": "state",
          "hand": 1,
          "phase": "PREFLOP",
          "pot": 30,
          "currentBet": 20,
          "cards": ["As", "Ah"],
          "communityCards": [],
          "currentPlayer": "player_id",
          "players": [...]
        }
        """
        if engine_state.get("type") != "state" or "state" not in engine_state:
            return engine_state
            
        state = engine_state["state"]
        table = state.get("table", {})
        players = table.get("players", [])
        to_act_idx = state.get("toActIdx", -1)
        
        # Find current player
        current_player = ""
        if 0 <= to_act_idx < len(players) and players[to_act_idx]:
            current_player = players[to_act_idx]["id"]
        
        # Find our player to get hole cards
        our_cards = []
        current_bet = 0
        for p in players:
            if p and p["id"] == self.player:
                cards = p.get("cards", [])
                # Convert card format: {"rank": "A", "suit": "HEART"} -> "Ah"
                for card in cards:
                    if card and "rank" in card and "suit" in card:
                        rank = card["rank"]
                        suit_map = {"HEART": "h", "DIAMOND": "d", "CLUB": "c", "SPADE": "s"}
                        suit = suit_map.get(card["suit"], "h")
                        our_cards.append(f"{rank}{suit}")
        
        # Convert community cards
        community_cards = []
        for card in state.get("board", []):
            if card and "rank" in card and "suit" in card:
                rank = card["rank"]
                suit_map = {"HEART": "h", "DIAMOND": "d", "CLUB": "c", "SPADE": "s"}
                suit = suit_map.get(card["suit"], "h")
                community_cards.append(f"{rank}{suit}")
        
        # Estimate current bet from player states (simplified)
        # In real game, you'd track betting history
        max_bet = 0
        for p in players:
            if p:
                # This is a simplification; actual bet tracking would be more complex
                max_bet = max(max_bet, 0)
        
        # Convert players format
        bot_players = []
        for i, p in enumerate(players):
            if p:
                bot_players.append({
                    "id": p["id"],
                    "chips": p.get("chips", 0),
                    "bet": 0,  # Engine doesn't expose this directly
                    "folded": False,  # Would need to track from actions
                    "position": i
                })
        
        return {
            "type": "state",
            "hand": 1,  # Engine doesn't track hand number
            "phase": state.get("phase", "WAITING"),
            "pot": state.get("pot", 0),
            "currentBet": current_bet,
            "cards": our_cards,
            "communityCards": community_cards,
            "currentPlayer": current_player,
            "players": bot_players
        }
    
    def translate_bot_action_to_engine(self, bot_action: dict) -> dict:
        """
        Translate bot action to competition engine format

        Bot sends:
        {"type": "action", "action": "call"}
        {"type": "action", "action": "raise", "amount": 50}

        Engine expects (EXACT FORMAT - lowercase actions!):
        {"type": "action", "action": "call"}
        {"type": "action", "action": "check"}
        {"type": "action", "action": "fold"}
        {"type": "action", "action": "raise", "amount": 50}
        """
        action = bot_action.get("action", "").lower()  # MUST BE LOWERCASE!

        engine_msg = {
            "type": "action",
            "action": action
        }

        # Add amount for raises
        if action == "raise" and "amount" in bot_action:
            engine_msg["amount"] = bot_action["amount"]

        return engine_msg
    
    async def run(self):
        """Main adapter loop"""
        url = self.get_ws_url()
        print(f"🤖 Connecting to competition engine at {url}")
        print(f"🎯 Player: {self.player} | Table: {self.table}")
        
        try:
            async with websockets.connect(url) as ws:
                # Join the table
                await ws.send(json.dumps({"type": "join"}))
                print(f"✅ Joined table as {self.player}")
                
                # Main game loop
                while True:
                    try:
                        msg = await ws.recv()
                        engine_state = json.loads(msg)
                        
                        # Translate state to bot format
                        bot_state = self.translate_state_to_bot(engine_state)
                        
                        # Handle state updates
                        if bot_state.get("type") == "state":
                            # Check if it's our turn
                            if bot_state.get("currentPlayer") == self.player:
                                # Let bot make decision
                                phase = bot_state.get("phase", "UNKNOWN")
                                pot = bot_state.get("pot", 0)
                                hand_num = bot_state.get("hand", 0)
                                
                                print(f"\n{'='*60}")
                                print(f"🎴 Hand #{hand_num} | Phase: {phase} | Pot: ${pot}")
                                print(f"🃏 Our cards: {bot_state.get('cards', [])}")
                                print(f"🎯 Community: {bot_state.get('communityCards', [])}")
                                
                                # Use bot's decision making
                                our_player = {"id": self.player, "chips": 1000, "bet": 0}
                                action = await self.bot.make_decision(
                                    bot_state, 
                                    our_player, 
                                    bot_state.get("players", [])
                                )
                                
                                if action:
                                    # Translate action to engine format
                                    engine_action = self.translate_bot_action_to_engine(action)
                                    await ws.send(json.dumps(engine_action))
                                    print(f"🎲 Action: {action}")
                                    print(f"{'='*60}\n")
                        
                        elif engine_state.get("type") == "error":
                            print(f"⚠️  Engine error: {engine_state.get('error')}")
                        
                    except websockets.exceptions.ConnectionClosed:
                        print("❌ Connection closed")
                        break
                        
        except Exception as e:
            print(f"❌ Connection error: {e}")
            raise


async def main():
    """Entry point"""
    import sys
    
    # Configuration
    API_KEY = sys.argv[1] if len(sys.argv) > 1 else "dev"
    TABLE = sys.argv[2] if len(sys.argv) > 2 else "table-1"
    PLAYER = sys.argv[3] if len(sys.argv) > 3 else "bot1"
    SERVER_URL = sys.argv[4] if len(sys.argv) > 4 else "ws://localhost:8080"
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║     🃏  COMPETITION POKER BOT ADAPTER  🃏              ║
║          Texas Hold'em Infrastructure Engine             ║
╚══════════════════════════════════════════════════════════╝
    """)

    # Competition engine adapter
    adapter = CompetitionAdapter(API_KEY, TABLE, PLAYER, SERVER_URL)

    try:
        await adapter.run()
    except KeyboardInterrupt:
        print("\n👋 Bot shutting down...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
