import numpy as np
from type import is_cycle_from_margins, type_from_margins

def prefs_to_margins(p):
    """
    Convert 6D preference vector (percentages, sum=100) to pairwise margins.
    
    p = [ABC, ACB, BAC, BCA, CAB, CBA]
    Returns m_ab, m_bc, m_ca in range [-1,1]
    """
    p = np.array(p) / 100.0  # normalize percentages to fractions
    m_ab = (p[0] + p[1] + p[4]) - (p[2] + p[3] + p[5])
    m_bc = (p[0] + p[2] + p[3]) - (p[1] + p[4] + p[5])
    m_ca = (p[3] + p[5] + p[4]) - (p[0] + p[1] + p[2])
    return m_ab, m_bc, m_ca


# Interactive loop
while True:
    choice = input("Enter ranking percentages (ABC ACB BAC BCA CAB CBA) or 'exit': ")
    if choice.strip().lower() == "exit":
        break
    
    try:
        prefs = list(map(float, choice.strip().split()))
        if len(prefs) != 6:
            raise ValueError
    except ValueError:
        print("Invalid input. Enter 6 numbers separated by spaces.")
        continue
    
    # Compute margins
    m_ab, m_bc, m_ca = prefs_to_margins(prefs)
    
    # Print results using functions from type.py
    print(f"Margins: m_ab={m_ab:.3f}, m_bc={m_bc:.3f}, m_ca={m_ca:.3f}")
    print("Cycle:", is_cycle_from_margins(m_ab, m_bc, m_ca))
    print("Type:", type_from_margins(m_ab, m_bc, m_ca))
