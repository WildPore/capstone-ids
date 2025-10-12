import argparse


def main():
    parser = argparse.ArgumentParser(
        description="An intrusion detection tool for the command line"
    )
    parser.add_argument("input_file", help="Path to input file")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose mode"
    )

    args = parser.parse_args()

    print(args.input_file)
    print(args.output)
    print(args.verbose)


if __name__ == "__main__":
    main()
