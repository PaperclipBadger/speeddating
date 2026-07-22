import hashlib

def round_function(x, round_num, key, half_bits):
    h = hashlib.sha256(key + round_num.to_bytes(1, 'big') + x.to_bytes((half_bits + 7)//8, 'big')).digest()
    return int.from_bytes(h, 'big') & ((1 << half_bits) - 1)

def feistel_round_apply(x, bits, key, rounds=4):
    half = bits // 2
    mask = (1 << half) - 1
    L = x >> half
    R = x & mask
    for i in range(rounds):
        L, R = R, L ^ round_function(R, i, key, half)
    return (L << half) | R

def feistel_round_unapply(x, bits, key, rounds=4):
    half = bits // 2
    mask = (1 << half) - 1
    L = x >> half
    R = x & mask
    for i in reversed(range(rounds)):
        R, L = L, R ^ round_function(L, i, key, half)
    return (L << half) | R

def bits_needed(m):
    n = 1
    while (1 << n) < m:
        n += 1
    return max(n, 2)  # need at least 2 bits for the L/R split

def permute(id_, M, key):
    bits = bits_needed(M)
    x = id_
    while True:
        x = feistel_round_apply(x, bits, key)
        if x < M:
            return x

def unpermute(x, M, key):
    bits = bits_needed(M)
    y = x
    while True:
        y = feistel_round_unapply(y, bits, key)
        if y < M:
            return y