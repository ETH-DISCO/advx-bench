import argparse

parser = argparse.ArgumentParser()
parser.add_argument("arg1", type=int, help="An argument", required=True)
args = parser.parse_args()

print("You successfully ran an array job with argument:", args.arg1)
