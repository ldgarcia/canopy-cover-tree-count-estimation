# Model configuration files
The configuration files are contained in this directory. They can be composed, meaning that keys defined in subsequent files override keys defined in previous files, while respecting the nested structure. For example, if we have the following configuration files:

`foo.yml`:

```yaml
settings:
    a: "a"
    b:
        c: "c"
    e: "e"
```

`bar.yml`:

```yaml
settings:
    a: "ax"
    b:
        c: "cx"
        d: "dx"
```

If we use `--config foo bar` the resulting configuration will be:

```yaml
settings:
    a: "ax"
    b:
        c: "cx"
        d: "dx"
    e: "e"
```

If we use `--config bar foo` the resulting configuration will be:

```yaml
settings:
    a: "a"
    b:
        c: "c"
        d: "dx"
    e: "e"
```

We use the PyYAML syntax extension, as described ![here](https://pyyaml.org/wiki/PyYAMLDocumentation).

The content of the configuration files must have the following structure (one or more of the namespaces below).

| Namespace | Nested | Use |  |  |
|-|-|-|-|-|
| `settings` | Yes | Define values for model settings (e.g. `BaseModelSettings`). |  |  |
| `slurm` | Yes | Define values for Slurm settings. |  |  |
| `model_name_suffix` | No | Define composable model name suffixes (joined by concatenation in the specified order of the config files). |  |  |
