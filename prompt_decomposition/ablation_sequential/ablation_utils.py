# ablation_utils.py
"""
Structure-Preserving Ablation Utilities

This module provides functions for creating ablated versions of Psych-101 prompts.
Ablation removes task-informative content while preserving choice markers (<<...>>),
enabling adversarial training to prevent shortcut learning.

Key principle: The model should NOT be able to predict choices confidently
when task information is removed. If it can, it's relying on choice history
patterns rather than understanding the task.

Example:
    Original: "You choose <<A>> and get 84.0 points."
    Ablated:  "You choose <<A>> and get some points."

Usage:
    from ablation_utils import ablate_text

    ablated_prompt = ablate_text(original_prompt)
"""

import re
from typing import List


# ============================================
# Qualitative Value Patterns
# ============================================

QUALITATIVE_VALUES = [
    'very little', 'a little', 'little',
    'average', 'medium', 'moderate',
    'a lot', 'much', 'very much', 'lots',
    'normal',
    'low', 'very low', 'extremely low', 'somewhat low',
    'high', 'very high', 'extremely high', 'somewhat high',
    'none', 'some', 'many', 'few',
    'small', 'large', 'big', 'tiny',
    'short', 'long',
    'fast', 'slow',
    'easy', 'hard', 'difficult',
    'good', 'bad', 'poor', 'excellent',
    'correct', 'incorrect', 'wrong', 'right',
]

QUALITATIVE_PATTERN = '|'.join(re.escape(v) for v in QUALITATIVE_VALUES)


# ============================================
# Line Removal Patterns
# ============================================

PATTERNS_TO_REMOVE = [
    r'^You will (?:be shown|view|see|encounter|play|perform|complete|receive)',
    r'^Your (?:task|goal|job|objective) is',
    r'^In this (?:task|experiment|game|study)',
    r'^This (?:task|experiment|game)',
    r'^(?:Press|Select|Choose|Click) the (?:corresponding|correct|appropriate)',
    r'^You (?:can|should|must|need to) (?:choose|select|press|respond|indicate|emit)',
    r'^(?:Each|Every) (?:trial|round|game|block)',
    r'^You will receive feedback',
    r'^Remember',
    r'^(?:Note|Notice|Keep in mind)',
    r'^After having',
    r'^When you',
    r'^If you',
    r'^If the',
    r'^The (?:goal|objective|aim|purpose)',
    r'^Points (?:will be|are)',
    r'^You (?:can|will) earn',
    r'^You can win or lose',
    r'^(?:There are|Each alien|Each planet|Machine \w)',
    r'^The (?:treasure|reward|outcome|spaceships?|planet)',
    r'^It is your choice',
    r'^Planet \w has',
    r'^Before you',
    r'^Both \w+ and \w+ can take',
    r'^\w+ can take \d+ values',
    r'^This feedback will',
    r'^You will be doing',
    r'^The digits are',
    r'^The items are',
    r'^The blue aliens',
    r'^The red aliens',
    r'^When you visit',
    r'^When you trade',
    r'^To visit a planet',
    r'^They have different',
    r'^Each rocket ship',
    r'^But sometimes',
    r'^How likely',
    r'^Whether you get',
    r'^If there is an alien',
    r'^On each trial',
    r'^There (?:is|are) \d+',
    r'^Turning over',
    r'^You can turn',
    r'^You can stop',
    r'^Machine \w will',
    r'^You can choose a slot',
    r'^When you select',
    r'^You will view a series',
    r'^After having recalled',
    r'^You will play \d+',
    r'^Each game will',
    r'^You need to respond',
    # Slot machine experiments (feng, sadeghiyeh, somerville, waltz, wilson)
    r'^You are participating in',
    r'^The two slot machines',
    r'^Each time you choose',
    r'^You choose a slot machine',
    r'^Each slot machine',
    r'^The first \d+ trials? in each game',
    r'^After these instructed trials',
    # frey2017cct
    r'^In each round',
    r'^Every card is',
    r'^In different rounds',
    r'^Loss and gain amounts',
    r'^You may (?:keep|also)',
    r'^Your gains and losses',
    r'^Press \w+ to (?:turn|flip|draw)',
    # lefebvre2017behavioural
    r'^You are going to visit',
    r'^Each casino owns',
    r'^You can play one of the machines',
]


# ============================================
# Core Functions
# ============================================

def should_remove_line(line: str) -> bool:
    """
    Check if a line should be removed entirely during ablation.

    Lines are removed if they contain task instructions or meta-information
    that doesn't include choice markers.

    Args:
        line: A single line of text from the prompt

    Returns:
        True if the line should be removed, False otherwise
    """
    stripped = line.strip()
    if not stripped:
        return False

    # Never remove lines with choice markers
    if '<<' in line and '>>' in line:
        return False

    for pattern in PATTERNS_TO_REMOVE:
        if re.match(pattern, stripped, re.IGNORECASE):
            return True

    return False


def ablate_line_content(line: str) -> str:
    """
    Apply content ablation patterns to a single line.

    Replaces specific values (points, positions, stimuli) with generic
    placeholders while preserving the sentence structure.

    Args:
        line: A single line of text

    Returns:
        The line with task-informative content replaced by placeholders
    """
    if not line.strip():
        return line

    # Dollar amounts (e.g., "You win 100.0$ and lose 0.0$" — steingroever IGT)
    line = re.sub(
        r'(win|lose)\s+-?\d+(?:\.\d+)?\$',
        r'\1 some money',
        line, flags=re.IGNORECASE
    )

    # Weather prediction feedback (speekenbrink)
    line = re.sub(
        r'You are (?:correct|wrong),\s*the weather is (?:indeed )?(?:fine|rainy)\.',
        'Feedback is given.',
        line, flags=re.IGNORECASE
    )

    # Card stimuli (speekenbrink)
    line = re.sub(
        r'You are seeing the following: (?:card \d+(?:, card \d+)*)\.',
        'You are seeing some cards.',
        line, flags=re.IGNORECASE
    )

    # Circle/square outcomes over stimuli (wise2019)
    line = re.sub(
        r'a (?:circle|square) is shown over stimulus (\w+)',
        r'an outcome is shown over stimulus \1',
        line, flags=re.IGNORECASE
    )

    # Shock delivery (wise2019)
    line = re.sub(
        r'Finally,\s*(?:no shocks are delivered|a shock is delivered for stimulus \w+|shocks are delivered for both stimulus \w+ and stimulus \w+)\.',
        'Finally, an outcome occurs.',
        line, flags=re.IGNORECASE
    )

    # Lottery descriptions (wulff2018description — handles optional comma before "or")
    line = re.sub(
        r'Lottery (\w+) offers -?\d+(?:\.\d+)?\s*points\s*with\s*\d+(?:\.\d+)?%\s*probability(?:,?\s*or\s*-?\d+(?:\.\d+)?\s*points\s*with\s*\d+(?:\.\d+)?%\s*probability)*\.',
        r'Lottery \1 has some outcomes.',
        line, flags=re.IGNORECASE
    )

    # Ball color in feedback (krueger) — "A black ball is chosen"
    line = re.sub(
        r'A (?:black|cyan|magenta|green|red|blue|yellow|white|silver|teal|orange|purple|turquoise|beige) ball is chosen',
        'A ball is chosen',
        line, flags=re.IGNORECASE
    )

    # Specific letters (e.g., "You see the letter A" → "You see a letter")
    line = re.sub(
        r'(You see|You saw|see) the letter [a-zA-Z](?![a-zA-Z])',
        r'\1 a letter',
        line, flags=re.IGNORECASE
    )

    # Encoded colors (e.g., "see colour1" → "see a stimulus")
    line = re.sub(
        r'(?:see|saw) colou?r\d+',
        'see a stimulus',
        line, flags=re.IGNORECASE
    )

    # Specific shapes
    line = re.sub(
        r'(?:see )?(?:a |the )?(?:big|small|large|tiny) (?:white|black|red|blue|green|yellow) (?:square|triangle|circle|rectangle)',
        'see a shape',
        line, flags=re.IGNORECASE
    )

    # Digit sequences
    line = re.sub(
        r'The (?:digits|numbers|items) are the following: \[[^\]]+\]',
        'Items are presented.',
        line, flags=re.IGNORECASE
    )

    # Option/lottery descriptions (plonsky/peterson — handles "either", ", or", decimals)
    line = re.sub(
        r'Option (\w) delivers (?:either )?-?\d+(?:\.\d+)?\s*points\s*with\s*(?:\d+(?:\.\d+)?%|unknown)\s*chance(?:,?\s*(?:or\s+)?-?\d+(?:\.\d+)?\s*points\s*with\s*(?:\d+(?:\.\d+)?%|unknown)\s*chance)*\.',
        r'Option \1 has some outcomes.',
        line, flags=re.IGNORECASE
    )
    line = re.sub(
        r'\d+(?:\.\d+)?%?\s*(?:chance|probability)\s*(?:of|to)[^,.)]+',
        'some chance of some outcome',
        line, flags=re.IGNORECASE
    )

    # Hazard rate (xiong2023neural: "The hazard rate is 0.2." → "The hazard rate is some probability.")
    line = re.sub(
        r'((?:T|t)he hazard rate is) \d+(?:\.\d+)?',
        r'\1 some probability',
        line
    )

    # Per-game trial count (xiong/feng: "There are 100 trials in this game.")
    line = re.sub(
        r'There are \d+ trials in this game',
        'There are some trials in this game',
        line, flags=re.IGNORECASE
    )

    # Per-round loss card count (frey: "There are 20 loss cards in this round.")
    line = re.sub(
        r'There are \d+ loss cards in this round',
        'There are some loss cards in this round',
        line, flags=re.IGNORECASE
    )

    # Per-environment choice count (wu: "You have 10 choices to make in this environment.")
    line = re.sub(
        r'You have \d+ choices to make',
        'You have some choices to make',
        line, flags=re.IGNORECASE
    )

    # Option value reveals (wu2018generalisation: "The value of option 21 is 70.")
    line = re.sub(
        r'The value of option \d+ is \d+',
        'The value of an option is revealed',
        line, flags=re.IGNORECASE
    )

    # Cue values (weather prediction, MCPL tasks)
    line = re.sub(
        rf'((?:Progladine|Amalydine|Caldionine|Geoplite|Soite|Cue \d+)):\s*(?:\d+(?:\.\d+)?|{QUALITATIVE_PATTERN})',
        r'\1: [value]',
        line, flags=re.IGNORECASE
    )

    # Ball/color counts (krueger — handles "There are X black balls, Y cyan balls, ...")
    line = re.sub(
        r'There are \d+ \w+ balls(?:,\s*\d+ \w+ balls)*(?:,?\s*and \d+ \w+ balls)?\.',
        'There are some colored balls.',
        line, flags=re.IGNORECASE
    )
    line = re.sub(
        r'\d+ (?:cyan|silver|teal|black|white|red|blue|green|yellow|orange|purple|magenta|turquoise|beige) balls?',
        'some balls',
        line, flags=re.IGNORECASE
    )

    # Card descriptions
    line = re.sub(
        r'(?:Card|Deck) (\w) (?:shows|has|contains|displays) [^.]+',
        r'Option \1 has some properties',
        line, flags=re.IGNORECASE
    )

    # Positions
    line = re.sub(r'at position \([^)]+\)', 'at a position', line, flags=re.IGNORECASE)

    # Alien/spaceship/planet names (two-step tasks)
    line = re.sub(r'alien (?:named |called )?[A-Z](?=[ ,.])', 'an alien', line)
    line = re.sub(r'aliens? (?:named )?[A-Z] and (?:alien )?[A-Z]', 'aliens', line)
    line = re.sub(r'spaceships? (?:called |named )?[A-Z](?: and [A-Z])?', 'spaceships', line)
    line = re.sub(r'(?:on |to )?(?:the )?(?:blue|red|green|yellow) planet', 'a planet', line)
    line = re.sub(r'planet (?:named |called )?[A-Z](?=[ ,.])', 'a planet', line)

    # Awarded points (frey2017cct: "You will be awarded 150 points for...")
    line = re.sub(
        r'(?:be )?awarded \d+(?:\.\d+)?\s*points',
        'be awarded some points',
        line, flags=re.IGNORECASE
    )

    # Running/final scores (frey2017cct: "Your current score is 150.")
    line = re.sub(
        r'(?:Your (?:current|final) score(?: for this round)? is) -?\d+(?:\.\d+)?',
        r'Your score is updated',
        line, flags=re.IGNORECASE
    )

    # Flesch gardening: tree attributes → "a tree"
    line = re.sub(
        r'You get a tree with level \d+ of leafiness and level \d+ of branchiness in the \w+ garden\.',
        'You see a tree.',
        line, flags=re.IGNORECASE
    )
    # Flesch: "You would have gotten 50 points, had you accepted/rejected to plant the tree."
    line = re.sub(
        r'You would have gotten -?\d+ points, had you (?:accepted|rejected) to plant the tree\.',
        'The correct answer is revealed.',
        line, flags=re.IGNORECASE
    )

    # Gershman2020reward: stimulus IDs → "a stimulus"
    line = re.sub(
        r'You see stimulus \d+\.',
        'You see a stimulus.',
        line, flags=re.IGNORECASE
    )

    # Hilbig: rating patterns → "some ratings"
    line = re.sub(
        r'Product (\w) ratings: \[[01 ]+\]\.',
        r'Product \1 ratings: [some ratings].',
        line, flags=re.IGNORECASE
    )

    # Levering: image codes → "an image"
    line = re.sub(
        r'(?:see the |see )image \d{3}',
        'see an image',
        line, flags=re.IGNORECASE
    )
    # Levering: typicality ratings
    line = re.sub(
        r'You rate the typicality as <<(\d+)>>',
        r'You rate the typicality as <<\1>>',
        line, flags=re.IGNORECASE
    )

    # Kool2016when: antimatter amounts
    line = re.sub(
        r'You find \d+ pieces? of antimatter\.',
        'You find an outcome.',
        line, flags=re.IGNORECASE
    )

    # Kool2017cost: treasure multiplier state
    line = re.sub(
        r'There is (?:no|a) treasure multiplier\.',
        'A condition is set.',
        line, flags=re.IGNORECASE
    )

    # Zorowitz: planet color → "a planet", alien references
    line = re.sub(
        r'You end up (?:on )?(?:the )?(?:blue|red) planet\.',
        'You end up on a planet.',
        line, flags=re.IGNORECASE
    )
    line = re.sub(
        r'You see (?:a )?(?:blue|red) (?:an )?alien (?:named \w+ )?and (?:a )?(?:blue|red) (?:an )?alien(?: named \w+)?',
        'You see two aliens',
        line, flags=re.IGNORECASE
    )
    line = re.sub(
        r'You (?:get|receive) (?:treasure|junk)\.',
        'You find an outcome.',
        line, flags=re.IGNORECASE
    )

    # Tomov2020discovery: station navigation → masked
    line = re.sub(
        r'The new starting station is \d+ and the goal station is \d+\.',
        'A new round begins.',
        line, flags=re.IGNORECASE
    )
    line = re.sub(
        r'Your station: \d+\. Neighboring stations: (?:\d+|circle) on the north, (?:\d+|circle) on the east, (?:\d+|circle) on the south, and (?:\d+|circle) on the west\.',
        'You are at a station.',
        line, flags=re.IGNORECASE
    )

    # Success/failure feedback (tomov2020discovery and similar)
    line = re.sub(
        r'You are successful\.',
        'An outcome is reached.',
        line, flags=re.IGNORECASE
    )

    # Tomov2021multitask: room/resource/price masking
    line = re.sub(
        r'The current market prices are -?\d+ for wood, -?\d+ for stone, and -?\d+ for iron\.',
        'New market prices are set.',
        line, flags=re.IGNORECASE
    )
    line = re.sub(
        r'You are in room \d+\.',
        'You are in a room.',
        line, flags=re.IGNORECASE
    )
    line = re.sub(
        r'and you find -?\d+ wood, -?\d+ stone, and -?\d+ iron\.',
        'and you find some resources.',
        line, flags=re.IGNORECASE
    )

    # Casino numbers (lefebvre: "You go to casino 3" → "You go to a casino")
    line = re.sub(
        r'(?:go to|visit) casino \d+',
        'go to a casino',
        line, flags=re.IGNORECASE
    )

    # Machine names (lefebvre: "between machines A and V" → "between machines")
    line = re.sub(
        r'(?:choose between|between) machines \w+ and \w+',
        'choose between machines',
        line, flags=re.IGNORECASE
    )

    # Option names (garcia: "between option T and option L" → "between options")
    line = re.sub(
        r'(?:choose between|between) option \w+ and option \w+',
        'choose between options',
        line, flags=re.IGNORECASE
    )

    # Specific points/rewards (includes "gain" for plonsky)
    line = re.sub(
        r'((?:and |you )?(?:get|receive|earn|win|lose|gain|got|received|gained|find|collect|accumulate))\s*-?\d+(?:\.\d+)?\s*(?:points?|pieces of space treasure)',
        r'\1 some points',
        line, flags=re.IGNORECASE
    )

    # Treasure outcomes
    line = re.sub(r'You find (?:treasure|junk)', 'You find an outcome', line, flags=re.IGNORECASE)

    # Correct answer feedback
    line = re.sub(
        rf'The correct (?:category|answer|response|concentration of \w+) (?:is|was) (?:indeed )?(?:{QUALITATIVE_PATTERN}|\w+)',
        'The correct answer is revealed',
        line, flags=re.IGNORECASE
    )

    # "That is correct/incorrect"
    line = re.sub(
        r'That is (?:in)?correct\.?\s*(?:The correct[^.]*\.)?',
        'Feedback is given.',
        line, flags=re.IGNORECASE
    )

    # Counterfactual information (includes "gained"/"lost" for plonsky)
    # Only replace the numeric value, preserving "had you chosen..." suffix
    line = re.sub(
        r'You would have (?:received|gotten|earned|gained|lost) -?\d+(?:\.\d+)?\s*points',
        'You would have received some points',
        line, flags=re.IGNORECASE
    )
    # Mask specific option references in counterfactuals
    line = re.sub(
        r'had you chosen (?:option \w+|the other option)(?: instead)?',
        'had you chosen a different option',
        line, flags=re.IGNORECASE
    )

    # Payoff information (use (?:[^.]|\.\d)+ to handle decimals)
    line = re.sub(
        r'The payoff[^.]*would be (?:[^.]|\.\d)+',
        'The payoff would be some points',
        line, flags=re.IGNORECASE
    )

    # Balloon outcomes (BART)
    line = re.sub(
        r'The balloon was inflated too much and explodes',
        'The outcome is revealed',
        line, flags=re.IGNORECASE
    )
    line = re.sub(
        r'You stop (?:inflating the balloon|pumping) and (?:get|collect) [^.]+',
        'You stop and collect points',
        line, flags=re.IGNORECASE
    )

    # Observations
    line = re.sub(
        r'and observe -?\d+(?:\.\d+)?\s*points',
        'and observe some points',
        line, flags=re.IGNORECASE
    )

    # Reaction times (remove entirely)
    line = re.sub(r'\s*in\s*-?\d+(?:\.\d+)?\s*(?:ms|milliseconds)\.?', '.', line, flags=re.IGNORECASE)
    line = re.sub(r'\s*RT:\s*\d+(?:\.\d+)?\s*(?:ms)?', '', line, flags=re.IGNORECASE)

    # Clean up artifacts
    line = re.sub(r'\.\.+', '.', line)
    line = re.sub(r'\s+\.', '.', line)
    line = re.sub(r'\s{2,}', ' ', line)

    return line


def extract_choices(text: str) -> List[str]:
    """
    Extract all choices from text.

    Args:
        text: Full prompt text

    Returns:
        List of all choice values found between << and >>
    """
    return re.findall(r'<<([^>]+)>>', text)


def get_unique_choice_options(text: str) -> List[str]:
    """
    Get unique choice options preserving their order of first appearance.

    Args:
        text: Full prompt text

    Returns:
        List of unique choice options in order of first appearance
    """
    choices = extract_choices(text)
    seen = set()
    unique = []
    for c in choices:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def create_minimal_framing(choice_options: List[str]) -> str:
    """
    Create minimal framing string for ablated prompt.

    The framing tells the model what response options are available
    without providing any task context.

    Args:
        choice_options: List of unique choice options

    Returns:
        A minimal framing string like "You will respond with A or B."
    """
    if len(choice_options) == 0:
        return "You will make choices."
    elif len(choice_options) == 1:
        return f"You will respond with {choice_options[0]}."
    elif len(choice_options) == 2:
        return f"You will respond with {choice_options[0]} or {choice_options[1]}."
    else:
        if len(choice_options) > 5:
            shown = choice_options[:4]
            return f"You will respond with {', '.join(shown)}, etc."
        options_str = ", ".join(choice_options[:-1]) + f", or {choice_options[-1]}"
        return f"You will respond with {options_str}."


def ablate_text(text: str) -> str:
    """
    Apply structure-preserving ablation to a full prompt.

    This is the main entry point for ablation. It:
    1. Extracts choice options to create minimal framing
    2. Removes instructional lines
    3. Ablates content in remaining lines
    4. Preserves all choice markers (<<...>>)

    Args:
        text: The original prompt text

    Returns:
        The ablated prompt with task information removed but choice
        history structure preserved

    Example:
        >>> original = '''You will play a game.
        ... You choose <<A>> and get 84.0 points.
        ... You choose <<B>> and get 12.0 points.'''
        >>> ablated = ablate_text(original)
        >>> print(ablated)
        You will respond with A or B.

        You choose <<A>> and get some points.
        You choose <<B>> and get some points.
    """
    choice_options = get_unique_choice_options(text)

    lines = text.split('\n')
    ablated_lines = []

    # Add minimal framing at the start
    framing = create_minimal_framing(choice_options)
    ablated_lines.append(framing)
    ablated_lines.append("")

    for line in lines:
        if should_remove_line(line):
            continue

        ablated_line = ablate_line_content(line)

        if ablated_line.strip():
            ablated_lines.append(ablated_line)
        elif ablated_lines and ablated_lines[-1].strip():
            ablated_lines.append("")

    # Clean up consecutive empty lines
    cleaned = []
    prev_empty = False
    for line in ablated_lines:
        if not line.strip():
            if not prev_empty:
                cleaned.append("")
            prev_empty = True
        else:
            cleaned.append(line)
            prev_empty = False

    return "\n".join(cleaned)


# ============================================
# Convenience Functions
# ============================================

def count_choices(text: str) -> int:
    """Count the number of choices in a prompt."""
    return len(extract_choices(text))


def get_ablation_ratio(original: str, ablated: str) -> float:
    """
    Calculate the character reduction ratio from ablation.

    Args:
        original: Original prompt text
        ablated: Ablated prompt text

    Returns:
        Ratio of characters removed (0.0 = no change, 1.0 = all removed)
    """
    if len(original) == 0:
        return 0.0
    return 1.0 - (len(ablated) / len(original))
