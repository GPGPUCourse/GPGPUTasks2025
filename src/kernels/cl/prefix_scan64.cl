#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#ifndef TILE
#define TILE 896
#endif
#ifndef MATRIX_HEIGHT
#define MATRIX_HEIGHT 64
#endif
#ifndef STATUS_FLAG_X
#define STATUS_FLAG_X 0
#endif

#define STATUS_FLAG_A 1
#define STATUS_FLAG_P 2
#define MATRIX_WIDTH (TILE % MATRIX_HEIGHT ? TILE / MATRIX_HEIGHT + 1 : TILE / MATRIX_HEIGHT)
#define MATRIX_SIZE (MATRIX_WIDTH * MATRIX_HEIGHT)

struct state {
    uint value;
    uint status;
};

union pun64 {
    struct state s;
    ulong u;
};

void upsweep(local uint* column) {
    uint local_id = get_local_id(0);

    for (uint offset = 1; offset < MATRIX_HEIGHT; offset *= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);

        uint mask = offset * 2 - 1;
        if ((local_id & mask) == mask) {
            column[local_id] += column[local_id - offset];
        }
    }
}

void downsweep(local uint* column) {
    uint local_id = get_local_id(0);

    for (uint offset = MATRIX_HEIGHT / 2; offset > 0; offset /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);

        uint mask = offset * 2 - 1;
        if ((local_id & mask) == mask) {
            uint temporary = column[local_id - offset];
            column[local_id - offset] = column[local_id];
            column[local_id] += temporary;
        }
    }
}

void scan_column(local uint* column) {
    upsweep(column);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) == 0) {
        column[MATRIX_HEIGHT - 1] = 0;
    }

    downsweep(column);
}

void matrix_scan(local uint* matrix, local uint* column) {
    uint local_id = get_local_id(0);

    if (local_id < MATRIX_HEIGHT) {
        local uint* row = matrix + local_id * MATRIX_WIDTH;
        uint result = 0;
        for (uint i = 0; i < MATRIX_WIDTH; i++) {
            result += row[i];
        }
        column[local_id] = result;
    }

    scan_column(column);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < MATRIX_HEIGHT) {
        local uint* row = matrix + local_id * MATRIX_WIDTH;
        uint prefix = column[local_id];
        for (uint i = 0; i < MATRIX_WIDTH; i++) {
            prefix += row[i];
            row[i] = prefix;
        }
    }
}

kernel void prefix_scan(
    global const uint* input,
    global uint* output,
    volatile global uint* global_counter,
    volatile global union pun64* states
) {
    local uint partition[MATRIX_SIZE];
    local uint column[MATRIX_HEIGHT];
    local uint global_id;

    uint local_id = get_local_id(0);

    if (local_id == 0) {
        global_id = atomic_inc(global_counter);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = 0; i < MATRIX_SIZE; i += MATRIX_HEIGHT) {
        uint j = i + local_id;
        if (j < TILE) {
            partition[j] = input[global_id * TILE + j];
        } else if (j < MATRIX_SIZE) {
            partition[j] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    matrix_scan(partition, column);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        union pun64 state;
        state.s.value = partition[MATRIX_SIZE - 1];
        state.s.status = STATUS_FLAG_A;
        atom_xchg(&states[global_id].u, state.u);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    volatile local int status_flag_p_index;
    local uint prefix;

    if (local_id == 0) {
        prefix = 0;
        status_flag_p_index = -1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint window_offset = 0; window_offset < global_id; window_offset += MATRIX_HEIGHT) {
        uint offset = window_offset + MATRIX_HEIGHT - local_id;
        column[local_id] = 0;

        if (offset <= global_id) {
            uint observed = global_id - offset;
            union pun64 state;
            do {
                state.u = atom_add(&states[observed].u, 0);
            } while (state.s.status == STATUS_FLAG_X);
            if (state.s.status == STATUS_FLAG_A) {
                column[local_id] = state.s.value;
            } else if (state.s.status == STATUS_FLAG_P) {
                column[local_id] = state.s.value;
                atomic_max(&status_flag_p_index, local_id);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        int flag = status_flag_p_index;
        if (flag >= 0 && local_id < flag) {
            column[local_id] = 0;
        }

        upsweep(column);

        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id == 0) {
            prefix += column[MATRIX_HEIGHT - 1];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (flag >= 0) {
            break;
        }
    }

    for (uint i = 0; i < TILE; i += MATRIX_HEIGHT) {
        uint j = i + local_id;
        if (j < TILE) {
            partition[j] += prefix;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        union pun64 state;
        state.s.value = partition[TILE - 1];
        state.s.status = STATUS_FLAG_P;
        atom_xchg(&states[global_id].u, state.u);
    }

    for (uint i = 0; i < TILE; i += MATRIX_HEIGHT) {
        uint j = i + local_id;
        if (j < TILE) {
            output[global_id * TILE + j] = partition[j];
        }
    }
}
