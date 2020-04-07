from constants import *


def error(msg):
    raise Exception(msg)


def get_sub_dict(arg):
    if len(arg) < 2:
        error("specify command")
    command = arg[1]

    if command not in flag_dict.keys():
        error(f"invalid command {command}")

    if command in ext_comm:
        sub_comm = arg[2]
        if sub_comm not in flag_dict[command].keys():
            error(f"incomplete command {command}")
        else:
            return command, sub_comm, flag_dict[command][sub_comm]

    else:
        return command, None, flag_dict[command]


def get_arg_dict(arg):
    command, sub_command, sub_dict = get_sub_dict(arg)
    arg_dict = {}

    for flag_name in sub_dict.keys():
        typ = sub_dict[flag_name]

        try:
            flag_idx = arg.index(flag_name)
        except ValueError:
            if sub_command is None:
                arg_dict[flag_name] = default_dict[command][flag_name]
            else:
                arg_dict[flag_name] = default_dict[command][sub_command][flag_name]
            continue

        if typ is None:
            if sub_command is None:
                arg_dict[flag_name] = not default_dict[command][flag_name]
            else:
                arg_dict[flag_name] = not default_dict[command][sub_command][flag_name]
        else:
            try:
                value = typ(arg[flag_idx + 1])
                if isinstance(value, Path):
                    value = value.absolute()
                arg_dict[flag_name] = value
            except ValueError or IndexError:
                error(f"invalid value for {arg[flag_idx]}")
                return
    return command, sub_command, arg_dict
