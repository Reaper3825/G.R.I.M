"""Compile supported Blender material nodes to the popup material IR."""

OP_LOAD_CONSTANT = 1
OP_LOAD_VERTEX_COLOR = 2
OP_LOAD_TEXCOORD = 3
OP_LOAD_WORLD_NORMAL = 4
OP_LAYER_WEIGHT = 7
OP_NOISE_TEXTURE = 9
OP_WAVE_TEXTURE = 10
OP_COLOR_RAMP = 11
OP_ADD = 12
OP_MULTIPLY = 13
OP_VECTOR_SCALE = 14
OP_MIX = 15
OP_EMISSION = 16
OP_TRANSPARENT = 17
OP_MIX_SHADER = 18
OP_OUTPUT_SURFACE = 19
OP_EXTRACT_COMPONENT = 20
OP_MAPPING = 21
OP_PRINCIPLED_LIT = 22

FLAG_CLAMP = 1 << 0
FLAG_CLAMP_FACTOR = 1 << 1
FLAG_CLAMP_RESULT = 1 << 2

MAX_REGISTERS = 64
MAX_INSTRUCTIONS = 64
MAX_PARAMETERS = 256


class MaterialCompiler:
    def __init__(self, material):
        if material is None or material.node_tree is None:
            raise RuntimeError("Popup material compiler requires a node material")
        self.material = material
        self.node_tree = material.node_tree
        self.instructions = []
        self.parameters = []
        self.next_register = 0
        self.output_cache = {}
        self.shared_cache = {}
        self.active_outputs = set()

    def compile(self):
        outputs = [node for node in self.node_tree.nodes
                   if node.type == "OUTPUT_MATERIAL" and node.is_active_output]
        if len(outputs) != 1:
            raise RuntimeError(
                f"Material '{self.material.name}' must have exactly one active Material Output"
            )
        surface = outputs[0].inputs.get("Surface")
        if surface is None or not surface.is_linked:
            raise RuntimeError(f"Material '{self.material.name}' has no linked Surface output")

        surface_register = self._compile_input(surface)
        destination = self._allocate_register()
        self._emit(OP_OUTPUT_SURFACE, destination, source_a=surface_register)
        return self.next_register, self.instructions, self.parameters

    def _allocate_register(self):
        if self.next_register >= MAX_REGISTERS:
            raise RuntimeError(
                f"Material '{self.material.name}' exceeds {MAX_REGISTERS} VM registers"
            )
        register = self.next_register
        self.next_register += 1
        return register

    def _add_parameters(self, values):
        offset = len(self.parameters)
        if offset + len(values) > MAX_PARAMETERS:
            raise RuntimeError(
                f"Material '{self.material.name}' exceeds {MAX_PARAMETERS} VM parameters"
            )
        self.parameters.extend(values)
        return offset, len(values)

    def _emit(self, opcode, destination, source_a=0, source_b=0, source_c=0,
              parameters=(), flags=0):
        if len(self.instructions) >= MAX_INSTRUCTIONS:
            raise RuntimeError(
                f"Material '{self.material.name}' exceeds {MAX_INSTRUCTIONS} VM instructions"
            )
        parameter_offset, parameter_count = self._add_parameters(parameters)
        self.instructions.append((opcode, destination, source_a, source_b, source_c,
                                  parameter_offset, parameter_count, flags))

    def _constant(self, value):
        if isinstance(value, (int, float)):
            packed = (float(value),) * 4
        else:
            components = tuple(float(component) for component in value)
            if len(components) == 3:
                packed = components + (0.0,)
            elif len(components) == 4:
                packed = components
            else:
                raise RuntimeError("Popup material constants must have 1, 3, or 4 components")
        destination = self._allocate_register()
        self._emit(OP_LOAD_CONSTANT, destination, parameters=(packed,))
        return destination

    def _compile_input(self, socket):
        if socket.is_linked:
            links = list(socket.links)
            if len(links) != 1:
                raise RuntimeError(
                    f"Material '{self.material.name}' input '{socket.name}' has multiple links"
                )
            link = links[0]
            return self._compile_output(link.from_node, link.from_socket)
        if not hasattr(socket, "default_value"):
            raise RuntimeError(
                f"Material '{self.material.name}' input '{socket.name}' has no value"
            )
        return self._constant(socket.default_value)

    def _compile_output(self, node, socket):
        key = (node.as_pointer(), socket.identifier)
        if key in self.output_cache:
            return self.output_cache[key]
        if key in self.active_outputs:
            raise RuntimeError(f"Material '{self.material.name}' contains a node cycle")
        self.active_outputs.add(key)
        try:
            register = self._dispatch_output(node, socket)
            self.output_cache[key] = register
            return register
        finally:
            self.active_outputs.remove(key)

    def _shared(self, node, key, build):
        cache_key = (node.as_pointer(), key)
        if cache_key not in self.shared_cache:
            self.shared_cache[cache_key] = build()
        return self.shared_cache[cache_key]

    def _extract(self, source, component):
        destination = self._allocate_register()
        self._emit(OP_EXTRACT_COMPONENT, destination, source_a=source,
                   parameters=((float(component), 0.0, 0.0, 0.0),))
        return destination

    def _dispatch_output(self, node, socket):
        handlers = {
            "VERTEX_COLOR": self._vertex_color,
            "TEX_COORD": self._texture_coordinate,
            "MAPPING": self._mapping,
            "LAYER_WEIGHT": self._layer_weight,
            "TEX_NOISE": self._noise_texture,
            "TEX_WAVE": self._wave_texture,
            "VALTORGB": self._color_ramp,
            "MATH": self._math,
            "VECT_MATH": self._vector_math,
            "MIX": self._mix,
            "EMISSION": self._emission,
            "BSDF_TRANSPARENT": self._transparent,
            "MIX_SHADER": self._mix_shader,
            "BSDF_PRINCIPLED": self._principled,
        }
        handler = handlers.get(node.type)
        if handler is None:
            raise RuntimeError(
                f"Material '{self.material.name}' uses unsupported node "
                f"'{node.name}' ({node.type})"
            )
        return handler(node, socket)

    def _vertex_color(self, node, socket):
        if not node.layer_name:
            raise RuntimeError(f"Material '{self.material.name}' has an unnamed Color Attribute")
        color = self._shared(node, "color", lambda: self._emit_source(OP_LOAD_VERTEX_COLOR))
        if socket.identifier == "Color":
            return color
        if socket.identifier == "Alpha":
            return self._extract(color, 3)
        raise RuntimeError(f"Unsupported Color Attribute output '{socket.name}'")

    def _emit_source(self, opcode):
        destination = self._allocate_register()
        self._emit(opcode, destination)
        return destination

    def _texture_coordinate(self, node, socket):
        if socket.identifier == "UV":
            return self._emit_source(OP_LOAD_TEXCOORD)
        if socket.identifier == "Normal":
            return self._emit_source(OP_LOAD_WORLD_NORMAL)
        raise RuntimeError(f"Unsupported Texture Coordinate output '{socket.name}'")

    def _mapping(self, node, socket):
        if socket.identifier != "Vector":
            raise RuntimeError(f"Unsupported Mapping output '{socket.name}'")
        for input_name in ("Location", "Rotation", "Scale"):
            if node.inputs[input_name].is_linked:
                raise RuntimeError(
                    f"Material '{self.material.name}' Mapping.{input_name} links are unsupported"
                )
        vector = self._compile_input(node.inputs["Vector"])
        parameters = tuple(
            tuple(float(component) for component in node.inputs[name].default_value) + (0.0,)
            for name in ("Location", "Rotation", "Scale")
        )
        destination = self._allocate_register()
        self._emit(OP_MAPPING, destination, source_a=vector, parameters=parameters)
        return destination

    def _layer_weight(self, node, socket):
        def build():
            normal = self._compile_input(node.inputs["Normal"])
            if node.inputs["Blend"].is_linked:
                raise RuntimeError("Linked Layer Weight blend is unsupported")
            blend = float(node.inputs["Blend"].default_value)
            destination = self._allocate_register()
            self._emit(OP_LAYER_WEIGHT, destination, source_a=normal,
                       parameters=((blend, 0.0, 0.0, 0.0),))
            return destination

        packed = self._shared(node, "fresnel-facing", build)
        if socket.identifier == "Fresnel":
            return self._extract(packed, 0)
        if socket.identifier == "Facing":
            return self._extract(packed, 1)
        raise RuntimeError(f"Unsupported Layer Weight output '{socket.name}'")

    def _noise_texture(self, node, socket):
        output_modes = {"Fac": 0.0, "Color": 1.0}
        if socket.identifier not in output_modes:
            raise RuntimeError(f"Unsupported Noise Texture output '{socket.name}'")
        dimensions = {"1D": 1.0, "2D": 2.0, "3D": 3.0, "4D": 4.0}
        if node.noise_dimensions not in dimensions:
            raise RuntimeError(f"Unsupported Noise Texture dimension '{node.noise_dimensions}'")
        scalar_inputs = list(node.inputs)[1:9]
        if len(scalar_inputs) != 8:
            raise RuntimeError(f"Unexpected Noise Texture socket layout on '{node.name}'")
        if any(socket.is_linked for socket in scalar_inputs):
            raise RuntimeError("Linked Noise Texture scalar inputs are unsupported")
        values = [float(socket.default_value) for socket in scalar_inputs]
        parameters = (
            (output_modes[socket.identifier], dimensions[node.noise_dimensions],
             1.0 if node.normalize else 0.0, 0.0),
            tuple(values[:4]),
            tuple(values[4:]),
        )
        vector = self._compile_input(node.inputs["Vector"])
        destination = self._allocate_register()
        self._emit(OP_NOISE_TEXTURE, destination, source_a=vector, parameters=parameters)
        return destination

    def _wave_texture(self, node, socket):
        output_modes = {"Color": 0.0, "Fac": 1.0}
        if socket.identifier not in output_modes:
            raise RuntimeError(f"Unsupported Wave Texture output '{socket.name}'")
        wave_types = {"BANDS": 0.0, "RINGS": 1.0}
        directions = {"X": 0.0, "Y": 1.0, "Z": 2.0, "DIAGONAL": 3.0,
                      "SPHERICAL": 4.0}
        profiles = {"SIN": 0.0, "SAW": 1.0, "TRI": 2.0}
        direction = node.bands_direction if node.wave_type == "BANDS" else node.rings_direction
        if (node.wave_type not in wave_types or direction not in directions or
                node.wave_profile not in profiles):
            raise RuntimeError(f"Unsupported Wave Texture mode on '{node.name}'")
        scalar_inputs = list(node.inputs)[1:7]
        if len(scalar_inputs) != 6:
            raise RuntimeError(f"Unexpected Wave Texture socket layout on '{node.name}'")
        if any(socket.is_linked for socket in scalar_inputs):
            raise RuntimeError("Linked Wave Texture scalar inputs are unsupported")
        values = [float(socket.default_value) for socket in scalar_inputs]
        parameters = (
            (output_modes[socket.identifier], wave_types[node.wave_type],
             directions[direction], profiles[node.wave_profile]),
            tuple(values[:4]),
            (values[4], values[5], 0.0, 0.0),
        )
        vector = self._compile_input(node.inputs["Vector"])
        destination = self._allocate_register()
        self._emit(OP_WAVE_TEXTURE, destination, source_a=vector, parameters=parameters)
        return destination

    def _color_ramp(self, node, socket):
        output_modes = {"Color": 0.0, "Alpha": 1.0}
        if socket.identifier not in output_modes:
            raise RuntimeError(f"Unsupported Color Ramp output '{socket.name}'")
        interpolation = {"CONSTANT": 0.0, "LINEAR": 1.0, "EASE": 2.0,
                         "B_SPLINE": 3.0, "CARDINAL": 4.0}
        ramp = node.color_ramp
        if ramp.interpolation not in interpolation or ramp.color_mode != "RGB":
            raise RuntimeError(f"Unsupported Color Ramp mode on '{node.name}'")
        parameters = [(output_modes[socket.identifier], interpolation[ramp.interpolation],
                       float(len(ramp.elements)), 0.0)]
        for element in ramp.elements:
            parameters.append((float(element.position), 0.0, 0.0, 0.0))
            parameters.append(tuple(float(component) for component in element.color))
        factor = self._compile_input(node.inputs["Fac"])
        destination = self._allocate_register()
        self._emit(OP_COLOR_RAMP, destination, source_a=factor, parameters=parameters)
        return destination

    def _math(self, node, socket):
        opcodes = {"ADD": OP_ADD, "MULTIPLY": OP_MULTIPLY}
        opcode = opcodes.get(node.operation)
        if opcode is None or socket.identifier != "Value":
            raise RuntimeError(f"Unsupported Math operation '{node.operation}'")
        source_a = self._compile_input(node.inputs[0])
        source_b = self._compile_input(node.inputs[1])
        destination = self._allocate_register()
        self._emit(opcode, destination, source_a=source_a, source_b=source_b,
                   flags=FLAG_CLAMP if node.use_clamp else 0)
        return destination

    def _vector_math(self, node, socket):
        if node.operation != "SCALE" or socket.identifier != "Vector":
            raise RuntimeError(f"Unsupported Vector Math operation '{node.operation}'")
        vector = self._compile_input(node.inputs[0])
        scale = self._compile_input(node.inputs[3])
        destination = self._allocate_register()
        self._emit(OP_VECTOR_SCALE, destination, source_a=vector, source_b=scale)
        return destination

    def _mix(self, node, socket):
        if node.data_type != "FLOAT" or socket.identifier != "Result_Float":
            raise RuntimeError(f"Unsupported Mix data type '{node.data_type}'")
        factor = self._compile_input(node.inputs[0])
        source_a = self._compile_input(node.inputs[2])
        source_b = self._compile_input(node.inputs[3])
        flags = 0
        if node.clamp_factor:
            flags |= FLAG_CLAMP_FACTOR
        if node.clamp_result:
            flags |= FLAG_CLAMP_RESULT
        destination = self._allocate_register()
        self._emit(OP_MIX, destination, source_a=factor,
                   source_b=source_a, source_c=source_b, flags=flags)
        return destination

    def _emission(self, node, socket):
        if socket.identifier != "Emission":
            raise RuntimeError(f"Unsupported Emission output '{socket.name}'")
        color = self._compile_input(node.inputs["Color"])
        strength = self._compile_input(node.inputs["Strength"])
        destination = self._allocate_register()
        self._emit(OP_EMISSION, destination, source_a=color, source_b=strength)
        return destination

    def _transparent(self, node, socket):
        if socket.identifier != "BSDF":
            raise RuntimeError(f"Unsupported Transparent BSDF output '{socket.name}'")
        color = self._compile_input(node.inputs["Color"])
        destination = self._allocate_register()
        self._emit(OP_TRANSPARENT, destination, source_a=color)
        return destination

    def _mix_shader(self, node, socket):
        if socket.identifier != "Shader":
            raise RuntimeError(f"Unsupported Mix Shader output '{socket.name}'")
        factor = self._compile_input(node.inputs[0])
        shader_a = self._compile_input(node.inputs[1])
        shader_b = self._compile_input(node.inputs[2])
        destination = self._allocate_register()
        self._emit(OP_MIX_SHADER, destination, source_a=factor,
                   source_b=shader_a, source_c=shader_b)
        return destination

    def _principled(self, node, socket):
        if socket.identifier != "BSDF":
            raise RuntimeError(f"Unsupported Principled output '{socket.name}'")
        base_color = node.inputs.get("Base Color")
        alpha = node.inputs.get("Alpha")
        normal = node.inputs.get("Normal")
        metallic = node.inputs.get("Metallic")
        roughness = node.inputs.get("Roughness")
        ior = node.inputs.get("IOR")
        emission_color = node.inputs.get("Emission Color")
        emission_strength = node.inputs.get("Emission Strength")
        required = (base_color, alpha, normal, metallic, roughness, ior,
                    emission_color, emission_strength)
        if any(input_socket is None for input_socket in required):
            raise RuntimeError("Principled BSDF socket layout is unsupported")
        constant_inputs = (metallic, roughness, ior, emission_color, emission_strength)
        if any(input_socket.is_linked for input_socket in constant_inputs):
            raise RuntimeError(
                "Linked Principled metallic, roughness, IOR, or emission inputs are unsupported"
            )

        color_register = self._compile_input(base_color)
        alpha_register = self._compile_input(alpha)
        normal_register = (self._compile_input(normal) if normal.is_linked
                           else self._emit_source(OP_LOAD_WORLD_NORMAL))
        parameters = (
            (float(metallic.default_value), float(roughness.default_value),
             float(ior.default_value), float(emission_strength.default_value)),
            tuple(float(component) for component in emission_color.default_value),
        )
        destination = self._allocate_register()
        self._emit(OP_PRINCIPLED_LIT, destination, source_a=color_register,
                   source_b=alpha_register, source_c=normal_register,
                   parameters=parameters)
        return destination


def compile_object_material_program(obj):
    materials = [slot.material for slot in obj.material_slots]
    if not materials or any(material is None for material in materials):
        raise RuntimeError(f"Object '{obj.name}' has an unassigned material slot")
    first = materials[0]
    if any(material.as_pointer() != first.as_pointer() for material in materials[1:]):
        raise RuntimeError(
            f"Object '{obj.name}' uses multiple material programs; split it before export"
        )
    return MaterialCompiler(first).compile()
