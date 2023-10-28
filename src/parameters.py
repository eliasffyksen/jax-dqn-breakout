import sys

def parse_parameter(name: str, required=True) -> str | None:
    value = list(filter(lambda x: x.startswith(f'--{name}='), sys.argv))

    if len(value) != 1:
        if required:
            print(f'you must specify one --{name}=<value>')
            exit(1)

        return None

    value = value[0][len(f'--{name}='):]

    return value
