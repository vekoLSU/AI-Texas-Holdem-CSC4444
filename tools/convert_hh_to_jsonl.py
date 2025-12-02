#!/usr/bin/env python3
"""
Convert PokerStars hand history files to JSONL format for replay in test_server.py

Usage:
    python3 tools/convert_hh_to_jsonl.py input.txt output.jsonl [hero_name] [--scale FACTOR]

Format:
    Each line in the output JSONL file represents one complete hand with all phases.
    Each hand contains: hand_id, hero_cards, phases (PREFLOP, FLOP, TURN, RIVER),
    and showdown information.
"""

import re
import json
import sys
import argparse
from typing import Dict, List, Optional, Tuple


def parse_card(card_str: str) -> List[str]:
    """Convert card string like 'Kd' to ['K', 'd']"""
    if len(card_str) == 2:
        return [card_str[0], card_str[1]]
    return ['?', '?']


def parse_cards(cards_str: str) -> List[List[str]]:
    """Parse multiple cards like '[Kd Kh]' to [['K','d'], ['K','h']]"""
    cards_str = cards_str.strip('[]')
    cards = cards_str.split()
    return [parse_card(card) for card in cards]


def extract_bet_amount(line: str, scale: float = 1.0) -> int:
    """Extract bet amount from action line and scale it"""
    # Look for patterns like "bets $20" or "raises $40 to $60"
    if 'to $' in line:
        match = re.search(r'to \$(\d+(?:\.\d+)?)', line)
        if match:
            return int(float(match.group(1)) * scale)
    elif 'bets $' in line or 'calls $' in line:
        match = re.search(r'\$(\d+(?:\.\d+)?)', line)
        if match:
            return int(float(match.group(1)) * scale)
    return 0


def parse_hand_history(hand_text: str, hero_name: str = "Hero", scale: float = 1.0) -> Optional[Dict]:
    """Parse a single hand history into structured format"""
    lines = hand_text.strip().split('\n')

    if not lines:
        return None

    # Extract hand ID
    hand_match = re.search(r'Hand #(\d+):', lines[0])
    if not hand_match:
        return None
    hand_id = hand_match.group(1)

    # Extract stakes and apply scaling
    stakes_match = re.search(r'\(\$(\d+(?:\.\d+)?)/\$(\d+(?:\.\d+)?)', lines[0])
    if stakes_match:
        small_blind = int(float(stakes_match.group(1)) * scale)
        big_blind = int(float(stakes_match.group(2)) * scale)
    else:
        small_blind = int(0.5 * scale)
        big_blind = int(1.0 * scale)

    # Parse players and starting stacks
    players = {}
    button_seat = 0

    for line in lines:
        # Extract button position
        button_match = re.search(r'Seat #(\d+) is the button', line)
        if button_match:
            button_seat = int(button_match.group(1))

        # Extract player info and apply scaling
        seat_match = re.search(r'Seat (\d+): (\w+) \(\$(\d+(?:\.\d+)?) in chips\)', line)
        if seat_match:
            seat_num = int(seat_match.group(1))
            player_name = seat_match.group(2)
            chips = int(float(seat_match.group(3)) * scale)
            players[player_name] = {
                'seat': seat_num,
                'chips': chips,
                'starting_chips': chips,
                'bet': 0,
                'folded': False
            }

    # Find hero's hole cards
    hero_cards = None
    for line in lines:
        if line.startswith('Dealt to ' + hero_name):
            cards_match = re.search(r'\[(.*?)\]', line)
            if cards_match:
                hero_cards = parse_cards(cards_match.group(1))

    if not hero_cards:
        return None

    # Parse betting rounds
    phases = {
        'PREFLOP': {'community_cards': [], 'pot': int(small_blind + big_blind), 'actions': []},
        'FLOP': {'community_cards': [], 'pot': 0, 'actions': []},
        'TURN': {'community_cards': [], 'pot': 0, 'actions': []},
        'RIVER': {'community_cards': [], 'pot': 0, 'actions': []}
    }

    current_phase = None

    # Track pot and bets
    pot = int(small_blind + big_blind)

    for line in lines:
        # Detect phase changes
        if '*** HOLE CARDS ***' in line:
            current_phase = 'PREFLOP'
        elif '*** FLOP ***' in line:
            current_phase = 'FLOP'
            cards_match = re.search(r'\[(.*?)\]', line)
            if cards_match:
                phases['FLOP']['community_cards'] = parse_cards(cards_match.group(1))
        elif '*** TURN ***' in line:
            current_phase = 'TURN'
            cards_match = re.search(r'\[(.*?)\]', line)
            if cards_match:
                phases['TURN']['community_cards'] = parse_cards(cards_match.group(1))
        elif '*** RIVER ***' in line:
            current_phase = 'RIVER'
            cards_match = re.search(r'\[(.*?)\]', line)
            if cards_match:
                phases['RIVER']['community_cards'] = parse_cards(cards_match.group(1))
        elif '*** SHOW DOWN ***' in line or '*** SUMMARY ***' in line:
            current_phase = None

        # Parse actions
        if current_phase and ':' in line:
            for player_name in players.keys():
                if line.startswith(player_name + ':'):
                    action_part = line.split(':', 1)[1].strip()

                    # Update player state based on action
                    if 'folds' in action_part:
                        players[player_name]['folded'] = True
                        phases[current_phase]['actions'].append({
                            'player': player_name,
                            'action': 'fold'
                        })
                    elif 'checks' in action_part:
                        phases[current_phase]['actions'].append({
                            'player': player_name,
                            'action': 'check'
                        })
                    elif 'calls' in action_part:
                        amount = extract_bet_amount(action_part, scale)
                        players[player_name]['chips'] -= amount
                        players[player_name]['bet'] += amount
                        pot += amount
                        phases[current_phase]['actions'].append({
                            'player': player_name,
                            'action': 'call',
                            'amount': amount
                        })
                    elif 'bets' in action_part or 'raises' in action_part:
                        amount = extract_bet_amount(action_part, scale)
                        players[player_name]['chips'] -= amount
                        players[player_name]['bet'] += amount
                        pot += amount
                        action_type = 'raise' if 'raises' in action_part else 'bet'
                        phases[current_phase]['actions'].append({
                            'player': player_name,
                            'action': action_type,
                            'amount': amount
                        })
                    elif 'posts small blind' in action_part:
                        players[player_name]['chips'] -= int(small_blind)
                        players[player_name]['bet'] = int(small_blind)
                    elif 'posts big blind' in action_part:
                        players[player_name]['chips'] -= int(big_blind)
                        players[player_name]['bet'] = int(big_blind)

    # Update pot for each phase
    running_pot = int(small_blind + big_blind)
    for phase_name in ['PREFLOP', 'FLOP', 'TURN', 'RIVER']:
        phases[phase_name]['pot'] = running_pot
        # Add up bets in this phase
        phase_bets = sum(action.get('amount', 0) for action in phases[phase_name]['actions'])
        running_pot += phase_bets

    # Parse showdown/winner info and apply scaling
    winner = None
    winning_amount = 0

    for line in lines:
        if 'collected' in line and '$' in line:
            winner_match = re.search(r'(\w+) collected \$(\d+(?:\.\d+)?)', line)
            if winner_match:
                winner = winner_match.group(1)
                winning_amount = int(float(winner_match.group(2)) * scale)
        elif ' won (' in line and '$' in line:
            winner_match = re.search(r'(\w+).*?won \(\$(\d+(?:\.\d+)?)\)', line)
            if winner_match:
                winner = winner_match.group(1)
                winning_amount = int(float(winner_match.group(2)) * scale)

    # Build player list for each phase
    result = {
        'hand_id': hand_id,
        'small_blind': int(small_blind),
        'big_blind': int(big_blind),
        'button_seat': button_seat,
        'hero_name': hero_name,
        'hero_cards': hero_cards,
        'players_initial': {name: info['starting_chips'] for name, info in players.items()},
        'phases': {},
        'showdown': {
            'winner': winner if winner else hero_name,
            'pot': running_pot,
            'final_board': phases['RIVER']['community_cards'] if phases['RIVER']['community_cards'] else []
        }
    }

    # Add state snapshot for each phase
    for phase_name in ['PREFLOP', 'FLOP', 'TURN', 'RIVER']:
        phase_data = phases[phase_name]

        # Build players list for this phase
        players_list = []
        position = 0
        for player_name in sorted(players.keys(), key=lambda p: players[p]['seat']):
            player_info = players[player_name].copy()
            players_list.append({
                'id': player_name,
                'chips': player_info['chips'],
                'bet': player_info['bet'],
                'folded': player_info['folded'],
                'position': position,
                'cards': hero_cards if player_name == hero_name else None
            })
            position += 1

        result['phases'][phase_name] = {
            'phase': phase_name,
            'pot': phase_data['pot'],
            'currentBet': max((p['bet'] for p in players_list), default=0),
            'communityCards': phase_data['community_cards'],
            'players': players_list,
            'actions': phase_data['actions']
        }

    return result


def convert_file(input_file: str, output_file: str, hero_name: str = "Hero", scale: float = 1.0):
    """Convert a PokerStars hand history file to JSONL format"""
    with open(input_file, 'r') as f:
        content = f.read()

    # Split into individual hands
    hands = re.split(r'\n\n+', content)

    converted_hands = []
    for hand_text in hands:
        if not hand_text.strip():
            continue

        hand_data = parse_hand_history(hand_text, hero_name, scale)
        if hand_data:
            converted_hands.append(hand_data)

    # Write to JSONL
    with open(output_file, 'w') as f:
        for hand in converted_hands:
            f.write(json.dumps(hand) + '\n')

    print(f"Converted {len(converted_hands)} hands from {input_file} to {output_file}")
    if scale != 1.0:
        print(f"Applied {scale}x scaling to all monetary amounts")


def main():
    parser = argparse.ArgumentParser(
        description='Convert PokerStars hand history files to JSONL format for replay',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python3 tools/convert_hh_to_jsonl.py input.txt output.jsonl

  # With custom hero name
  python3 tools/convert_hh_to_jsonl.py input.txt output.jsonl --hero MyName

  # Scale $0.50/$1 to $10/$20 (20x scaling for whole dollar game)
  python3 tools/convert_hh_to_jsonl.py input.txt output.jsonl --scale 20

  # Scale $0.50/$1 to $5/$10 (10x scaling)
  python3 tools/convert_hh_to_jsonl.py input.txt output.jsonl --scale 10
        """
    )

    parser.add_argument('input_file', help='Input PokerStars hand history file (.txt)')
    parser.add_argument('output_file', help='Output JSONL file')
    parser.add_argument('--hero', default='Hero',
                       help='Player name to track as hero (default: Hero)')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Multiply all monetary amounts by this factor (default: 1.0). '
                            'Use 10 to convert $0.50/$1 to $5/$10, or 20 for $10/$20')

    args = parser.parse_args()

    convert_file(args.input_file, args.output_file, args.hero, args.scale)


if __name__ == '__main__':
    main()
