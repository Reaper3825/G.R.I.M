# Host input support

Native tool-driven device execution belongs to [input_plugin](../../plugins/input_plugin/README.md).
This directory retains host UI keybindings, cursor presentation, pure observation
targeting geometry and checked-click helpers. It contains no host input tool service.
Perception is an independent provider; `click_on` and the old context-manager
integration remain retired.
