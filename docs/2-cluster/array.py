import sys

if len(sys.argv) != 2:
    print("Usage: python array.py arg1")
    print(len(sys.argv))
    sys.exit(1)

arg1 = int(sys.argv[1])
print("You successfully ran an array job with argument:", arg1)

