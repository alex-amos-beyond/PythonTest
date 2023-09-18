from PythonTest import divide3, times2


def AssembleCalcs(a):

    b1 = times2(a)
    b2 = divide3(b1)
    print(b2)

    return b2


if __name__ == "__main__":
    rslt = AssembleCalcs(6)