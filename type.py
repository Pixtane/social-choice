def is_cycle_from_margins(m_ab, m_bc, m_ca):
    """
    Determine if a preference profile is cyclic (Condorcet cycle)
    based on pairwise margins.
    
    Returns:
        True  -> cyclic
        False -> acyclic
    """
    # Check for existence of Condorcet winner
    # A winner exists if any alternative beats both others
    if (m_ab > 0 and m_ca < 0) or \
       (m_bc > 0 and m_ab < 0) or \
       (m_ca > 0 and m_bc < 0):
        return False  # acyclic
    else:
        return True   # cyclic


def type_from_margins(m_ab, m_bc, m_ca):
    """
    Determine preference type from pairwise margins.
    
    Returns:
        - One of "ABC", "ACB", "BAC", "BCA", "CAB", "CBA" for acyclic
        - "Cyclic" if no Condorcet winner
    """
    if is_cycle_from_margins(m_ab, m_bc, m_ca):
        return "Cyclic"
    
    # Determine ranking directly from margins
    # Start with a list of alternatives
    ranking = []
    
    # Determine who is first
    if m_ab > 0 and m_ca < 0:
        ranking = ['A','B','C'] if m_bc > 0 else ['A','C','B']
    elif m_bc > 0 and m_ab < 0:
        ranking = ['B','C','A'] if m_ca > 0 else ['B','A','C']
    elif m_ca > 0 and m_bc < 0:
        ranking = ['C','A','B'] if m_ab > 0 else ['C','B','A']
    else:
        # Fallback (should not happen)
        return "Unknown"
    
    return ''.join(ranking)


if __name__ == "__main__":
    # Example interactive loop
    while True:
        choice = input("Enter m_ab, m_bc, m_ca (or 'exit'): ")
        if choice.strip().lower() == "exit":
            break
        try:
            m_ab, m_bc, m_ca = map(float, choice.strip().split())
        except ValueError:
            print("Invalid input, enter three numbers separated by spaces")
            continue
        
        print("Margins:", m_ab, m_bc, m_ca)
        print("Cycle:", is_cycle_from_margins(m_ab, m_bc, m_ca))
        print("Type:", type_from_margins(m_ab, m_bc, m_ca))
