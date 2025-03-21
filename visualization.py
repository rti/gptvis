import pyvista as pv
import numpy as np
import matplotlib


def visualize_vector_as_cubes(
    vector,
    plotter,
    cmap,
    base_scale=0.1,
    scale_factor=0.9,
    origin=(0.0, 0.0, 0.0),
    neutral_fill_value=0.0,
    label=None,
    min_scale=None,
    max_scale=None,
):
    if not isinstance(vector, np.ndarray):
        raise TypeError(f"Input vector must be a NumPy array, got {type(vector)}")
    if vector.ndim != 1:
        raise ValueError(
            f"Input vector must be 1-dimensional, got {vector.ndim} dimensions."
        )

    vector_size = vector.shape[0]
    if vector_size == 0:
        raise ValueError("Input vector is empty. Nothing to visualize.")

    elements_per_cube = 4
    remainder = vector_size % elements_per_cube
    padded_vector = vector
    if remainder != 0:
        padding_size = elements_per_cube - remainder
        padding = np.full(padding_size, neutral_fill_value, dtype=vector.dtype)
        padded_vector = np.concatenate((vector, padding))

    num_cubes = len(padded_vector) // elements_per_cube

    data = padded_vector.reshape((num_cubes, elements_per_cube))

    scale_data = data[:, :3]
    color_data = data[:, 3]

    if min_scale is None or max_scale is None:
        min_scale, max_scale = scale_data.min(), scale_data.max()

    # print("#####", min_scale, max_scale)

    if max_scale == min_scale:
        normalized_scales = np.full_like(
            scale_data, base_scale + scale_factor / 2.0
        )  # Mid-range scale
    else:
        # Normalize to [0, 1] first
        norm_01 = (scale_data - min_scale) / (max_scale - min_scale)
        # Then scale to [base_scale, base_scale + scale_factor]
        normalized_scales = base_scale + norm_01 * scale_factor
        # print(normalized_scales)

    # Normalize color data to [0, 1]
    min_color, max_color = color_data.min(), color_data.max()
    print(f"{min_color} {max_color}")
    if max_color == min_color:
        normalized_colors = np.full_like(color_data, 0.5)  # Mid-range grey
    else:
        normalized_colors = (color_data - min_color) / (max_color - min_color)

    meshes = []

    z_cumulative = 0

    for i in range(num_cubes):
        scale_x, scale_y, scale_z = normalized_scales[i]
        color = normalized_colors[i]

        center_x = origin[0] + 0
        center_y = origin[1] + 0
        center_z = origin[2] + z_cumulative + (scale_z / 2.0)

        # Create the cube
        cube = pv.Cube(
            center=(center_x, center_y, center_z),
            x_length=scale_x,
            y_length=scale_y,
            z_length=scale_z,
        )

        if not isinstance(cube, pv.DataSet):
            raise ValueError("cannot create cube")

        mesh = plotter.add_mesh(
            cube,
            color=cmap(color),
            show_edges=True,  # not exported to gltf
        )
        meshes.append(mesh)

        z_cumulative += scale_z

    if label:
        plotter.add_mesh(
            pv.Text3D(
                label,
                height=0.4,
                depth=0.05,
                center=(origin[0], origin[1], origin[2] - 0.5),
                normal=(0, -1, 0),
            ),
            color=cmap(0),
        )

    return meshes


def view_transformer_and_attention(
    vis_data, input_token_texts, filename=None, attention_threshold=0.5
):
    plotter = pv.Plotter(off_screen=(filename is not None), window_size=(1920, 1080))
    if plotter.camera is None:
        raise ValueError("plotter has no camera")

    cmap = matplotlib.colormaps.get_cmap(
        "magma"
    )  # Or 'plasma', 'magma', 'coolwarm', etc.

    x_spacing = 3.5  # Horizontal space between tokens
    y_spacing = 5.0  # Vertical space between layers
    attention_head_offset = 0.4  # Small offset for attention lines of different heads
    plotter_y_offset = 0

    scale_data = []
    for i in vis_data.embeddings:
        scale_data += [*i]
    for i in vis_data.transformer_blocks:
        for j in i:
            scale_data += [*j]
    # print(scale_data)
    # min_scale = -0.5
    # max_scale = 0.5
    # print(min_scale, max_scale)

    if vis_data.embeddings is not None and len(vis_data.embeddings) > 0:
        pos_x = 0
        for i, embedding in enumerate(vis_data.embeddings):
            visualize_vector_as_cubes(
                embedding,
                cmap=cmap,
                label=input_token_texts[i],
                origin=(pos_x, plotter_y_offset, 0.0),
                plotter=plotter,
                # min_scale=min_scale,
                # max_scale=max_scale,
            )
            pos_x += x_spacing

        label_pos = (pos_x - x_spacing + 1.5, plotter_y_offset, 0)
        plotter.add_mesh(
            pv.Text3D(
                f"Embedding",
                height=0.3,
                depth=0.05,
                center=label_pos,
                normal=(0, -1, 0),
            ),
            color=cmap(0),
        )

        plotter_y_offset += y_spacing

    token_positions = []
    for i, layer in enumerate(vis_data.transformer_blocks):
        pos_x = 0
        seq_len = layer.shape[0]
        layer_token_positions = []
        for token_idx in range(seq_len):
            token_vector = layer[token_idx]
            token_position = (pos_x, plotter_y_offset, 0.0)
            visualize_vector_as_cubes(
                token_vector,
                cmap=cmap,
                plotter=plotter,
                origin=token_position,
                # min_scale=min_scale,
                # max_scale=max_scale,
            )
            # Add arrow pointing upwards from below the token
            arrow_start = (
                token_position[0],
                token_position[1] - y_spacing,
                0.24,
            )  # Start below the cubes

            arrow = pv.Arrow(
                    start=arrow_start,
                    direction=(0, 1, 0),
                    scale=y_spacing * 0.95,  # Length towards the next layer
                    shaft_radius=0.02,
                    tip_length=0.1,
                    tip_radius=0.05,
                )

            arrow = arrow.scale([1.0, 1.0, 0.01])


            plotter.add_mesh(
                arrow,
                color=cmap(0.9999),
            )
            layer_token_positions.append(token_position)
            pos_x += x_spacing
        token_positions.append(layer_token_positions)

        label_pos = (pos_x - x_spacing + 2.5, plotter_y_offset, 0)
        plotter.add_mesh(
            pv.Text3D(
                f"Transformer Layer {i+1}",
                height=0.3,
                depth=0.05,
                center=label_pos,
                normal=(0, -1, 0),
            ),
            color=cmap(1),
        )
        plotter_y_offset += y_spacing  # Move up for the next layer

    for i, attention_data in enumerate(vis_data.attention_data):
        layer_token_positions = token_positions[i]
        num_heads = len(attention_data)

        for head_idx, attention_matrix in enumerate(attention_data):
            for i in range(
                len(layer_token_positions)
            ):  # Query token (receiving attention)
                for j in range(
                    len(layer_token_positions)
                ):  # Key token (providing attention)
                    # Causal mask check: only draw if query attends to key or itself (j <= i)
                    if j > i:
                        continue

                    # Do not draw self-attention
                    if j == i:
                        continue

                    weight = attention_matrix[i, j]

                    if weight > attention_threshold:
                        # Start point (key token j) with offset
                        start_pos = list(layer_token_positions[j])

                        # End point (query token i) with offset
                        end_pos = list(layer_token_positions[i])

                        mid_pos = [
                            (start_pos[0] + ((end_pos[0] - start_pos[0]) * (3 / 4))),
                            start_pos[1],
                            (
                                ((start_pos[2] + end_pos[2]) / 2)
                                - 0.5
                                - (0.2 * head_idx)
                            ),
                        ]

                        color = cmap(head_idx / num_heads)
                        line_width = max(1, weight * 3.0)

                        plotter.add_mesh(
                            pv.Spline(
                                np.array([start_pos, mid_pos, end_pos]),
                                n_points=32,
                            ).tube(line_width * 0.01),
                            color=color,
                            point_size=None,
                        )

    plotter.camera_position = "xz"
    plotter.camera.azimuth = 30
    plotter.camera.elevation = 25
    plotter.reset_camera(plotter, bounds=plotter.bounds)
    plotter.camera.zoom(1.4)

    plotter.export_gltf("scene.gltf")

    if filename:
        plotter.screenshot(filename)
        plotter.close()
    else:
        plotter.show()
