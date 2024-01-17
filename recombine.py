def recombine(input_files: list[str], output_file: str):
    with open(input_files[0], "r") as f:
        text = f.read()
    for fp in input_files[1:]:
        with open(fp, "r") as f:
            text = text.replace(f"#include \"{fp}\"\n", f.read())

    with open(output_file, "w") as f:
        f.write(text)


if __name__ == "__main__":
    recombine([
        "main.cpp",
        "net.hpp",
        "layer.hpp",
        "matrix.hpp",
        "general_includes.hpp",
    ],
        "main_full.cpp"
    )