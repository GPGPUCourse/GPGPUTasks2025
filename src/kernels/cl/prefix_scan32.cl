#ifndef TILE
#define TILE 896
#endif
#ifndef MATRIX_HEIGHT
#define MATRIX_HEIGHT 64
#endif

#define MATRIX_WIDTH (TILE % MATRIX_HEIGHT ? TILE / MATRIX_HEIGHT + 1 : TILE / MATRIX_HEIGHT)
#define MATRIX_SIZE (MATRIX_WIDTH * MATRIX_HEIGHT)

union pun {
    uint u;
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

kernel void
partition_scan(global uint* data, volatile global uint* global_counter, global uint* aggregates) {
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
            partition[j] = data[global_id * TILE + j];
        } else if (j < MATRIX_SIZE) {
            partition[j] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    matrix_scan(partition, column);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        aggregates[global_id] = partition[MATRIX_SIZE - 1];
    }

    for (uint i = 0; i < TILE; i += MATRIX_HEIGHT) {
        uint j = i + local_id;
        if (j < TILE) {
            data[global_id * TILE + j] = partition[j];
        }
    }
}

kernel void global_scan(
    global uint* data,
    volatile global uint* global_counter,
    global uint* aggregates,
    volatile global union pun* inclusive_prefixes
) {
    local uint partition[MATRIX_SIZE];
    local uint column[MATRIX_HEIGHT];
    local uint global_id;
    uint local_id = get_local_id(0);

    column[local_id] = 0;

    if (local_id == 0) {
        global_id = atomic_inc(global_counter);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = 0; i < MATRIX_SIZE; i += MATRIX_HEIGHT) {
        uint j = i + local_id;
        if (j < TILE) {
            partition[j] = data[global_id * TILE + j];
        } else if (j < MATRIX_SIZE) {
            partition[j] = 0;
        }
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
            union pun pun;
            pun.u = atomic_add(&inclusive_prefixes[observed].u, 0);
            if (pun.u == (uint)-1) {
                column[local_id] = aggregates[observed];
            } else {
                column[local_id] = pun.u;
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
        atomic_xchg(&inclusive_prefixes[global_id].u, partition[TILE - 1]);
    }

    for (uint i = 0; i < TILE; i += MATRIX_HEIGHT) {
        uint j = i + local_id;
        if (j < TILE) {
            data[global_id * TILE + j] = partition[j];
        }
    }
}

