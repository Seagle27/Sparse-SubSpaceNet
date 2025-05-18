import numpy as np


def get_array_locations(array_form: str):
    array_type = array_form.lower()
    if array_type.startswith("mra"):
        if "4" in array_type:
            return np.array([0, 1, 4, 6])
        if "5" in array_type:
            return np.array([0, 1, 4, 7, 9])
        elif "6" in array_type:
            return np.array([0, 1, 6, 9, 11, 13])
        elif "7" in array_type:
            return np.array([0, 1, 4, 10, 12, 15, 17])
        elif "8" in array_type:
            return np.array([0, 1, 4, 10, 16, 18, 21, 23])
        else:
            raise Exception(f"{array_type} isn't supported")
    if array_type.startswith("coprime"):
        try:
            _, n_str, m_str = array_type.split('_')
            return coprime_array(int(n_str), int(m_str))
        except (ValueError, AttributeError):
            raise ValueError(
                f'"{array_type}" is not in the expected "coprime_<N>_<M>" format.'
            )
    else:
        raise Exception(f"{array_type} isn't supported")


def get_virtual_array(array_loc):
    array_locations = np.array(array_loc, dtype=float)  # Ensure numpy array
    coarray = array_locations[:, None] - array_locations[None, :]  # Difference coarray
    unique_lags = np.sort(np.unique(coarray))
    return unique_lags[unique_lags > 0]


def get_virtual_ula_array(array_loc):
    unique_lags = get_virtual_array(array_loc)
    largest_ula_element = 0

    for i in range(1, len(unique_lags) + 1):
        if i not in unique_lags:
            break
        largest_ula_element = i

    return np.arange(0, largest_ula_element + 1, 1)

def coprime_array(N: int, M: int) -> np.ndarray:
    r"""
    Generate the sensor locations (in half‐wavelength units) of a *coprime sparse array*
    formed by interleaving two uniform linear sub‑arrays:

    * Sub‑arrayA: `N` sensors with spacing `M·d`
      → positions{0,M,2M,…,(N−1)M\}.
    * Sub‑arrayB: `2M−1` sensors with spacing `N·d`
      → positions{N,2N,…,(2M−1)N}.

    The two integers `N` and `M` **must be coprime**.

    Parameters
    ----------
    N, M : int
        Coprime integers (e.g.N=4,M=3).

    Returns
    -------
    np.ndarray
        Sorted 1‑D array of unique sensor indices (integer multiples of
        :math:`d = \lambda / 2`).  Length is `N + 2M − 1`.

    Notes
    -----
    The physical spacing *d* is absent from the output because most array
    processing formulas use sensor coordinates normalised by *d*.  Multiply
    by your actual half‑wavelength distance later if needed.
    """
    # basic validation
    if np.gcd(N, M) != 1:
        raise ValueError(f"N={N} and M={M} are not coprime.")

    # sub‑array A positions: 0, M, 2M, …, (N‑1)M
    a_positions = np.arange(N) * M

    # sub‑array B positions: N, 2N, …, (2M‑1)N
    b_positions = np.arange(1, 2 * M) * N

    # merge, drop duplicates (none by design), sort
    pos = np.concatenate((a_positions, b_positions))
    indexes = np.unique(pos, return_index=True)[1]
    positions = np.array([pos[index] for index in sorted(indexes)])

    return positions


