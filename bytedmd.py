import dis
import math


def measureDMD(func, *args):
    """Measure ByteDMD cost of running func with given arguments.

    Simulates execution on an idealized processor with an LRU stack.
    Cost of reading data at stack distance d is sqrt(d).
    """
    code = func.__code__
    arg_names = code.co_varnames[:code.co_argcount]

    # Map variable names to byte sizes
    var_sizes = {}
    for name, val in zip(arg_names, args):
        var_sizes[name] = val.nbytes if hasattr(val, 'nbytes') else 1

    # Initialize LRU stack: first arg at bottom, last arg at top
    # Each entry is (name, byte_index)
    lru_stack = []
    for name in arg_names:
        for i in range(var_sizes[name]):
            lru_stack.append((name, i))

    cost = 0.0
    pending_reads = []

    # Python value stack to track operand names
    value_stack = []
    result_counter = 0

    # Opcodes that consume two operands and produce a result
    binary_opcodes = {
        'BINARY_ADD', 'BINARY_SUBTRACT', 'BINARY_MULTIPLY',
        'BINARY_TRUE_DIVIDE', 'BINARY_FLOOR_DIVIDE', 'BINARY_MODULO',
        'BINARY_POWER', 'BINARY_OP', 'COMPARE_OP',
    }

    for instr in dis.get_instructions(func):
        if instr.opname == 'LOAD_FAST':
            var_name = instr.argval
            if var_name in var_sizes:
                for byte_idx in range(var_sizes[var_name]):
                    key = (var_name, byte_idx)
                    distance = len(lru_stack) - lru_stack.index(key)
                    cost += math.sqrt(distance)
                pending_reads.append(var_name)
            value_stack.append(var_name)

        elif instr.opname in binary_opcodes:
            # LRU update: move accessed variables to top in access order
            for var_name in pending_reads:
                for byte_idx in range(var_sizes[var_name]):
                    key = (var_name, byte_idx)
                    lru_stack.remove(key)
                    lru_stack.append(key)
            pending_reads = []

            # Pop operands, determine result size
            right = value_stack.pop()
            left = value_stack.pop()
            if instr.opname == 'COMPARE_OP':
                result_size = 1  # boolean
            else:
                result_size = max(var_sizes.get(left, 1), var_sizes.get(right, 1))

            result_name = f"_r{result_counter}"
            result_counter += 1
            var_sizes[result_name] = result_size
            for i in range(result_size):
                lru_stack.append((result_name, i))
            value_stack.append(result_name)

        elif instr.opname == 'STORE_FAST':
            # Rename the result on the LRU stack to the variable name
            source_name = value_stack.pop()
            stored_name = instr.argval
            size = var_sizes[source_name]
            var_sizes[stored_name] = size
            for i in range(size):
                idx = lru_stack.index((source_name, i))
                lru_stack[idx] = (stored_name, i)
            del var_sizes[source_name]

    return cost
