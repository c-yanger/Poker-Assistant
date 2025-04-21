from deuces import Card, Evaluator

# Function to parse the input string of cards into Card objects
def parse_card_input(input_str):
    parts = input_str.strip().split()
    cards = []
    for p in parts:
        try:
            if len(p) >= 2:  # Ensure the card string is valid (at least 2 characters)
                cards.append(Card.new(p))
        except:
            print(f"⚠️ Skipping invalid card: {p}")
    return cards

def main():
    evaluator = Evaluator()

    # Get the player's hand (2 cards)
    print("Enter your hand (e.g., 'Ah Ks'):")
    hand_input = input("> ")
    hand = parse_card_input(hand_input)

    # Get the community cards (5 cards)
    print("Enter community cards (e.g., '2h 7d 9s Qh Tc'):")
    board_input = input("> ")
    board = parse_card_input(board_input)

    # Evaluate the hand using the evaluator
    score = evaluator.evaluate(hand, board)
    rank_class = evaluator.get_rank_class(score)

    # Print the result of the hand evaluation
    print("\n--- Result ---")
    print("Raw score:", score)
    print("Hand type:", evaluator.class_to_string(rank_class))

    # Display the player's hand
    print("\nYour hand:")
    Card.print_pretty_cards(hand)

    # Display the community cards
    print("Board:")
    Card.print_pretty_cards(board)

if __name__ == "__main__":
    main()
