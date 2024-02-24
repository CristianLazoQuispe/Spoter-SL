#!/bin/bash

# Función para generar todas las combinaciones de parámetros
generate_combinations() {
    local prefix=$1
    shift
    local values=("$@")

    for i in "${values[@]}"; do
        echo "$prefix: $i"
    done
}

# Función para ejecutar el script Python con las combinaciones de parámetros
execute_python() {
    local yaml_file=$1
    local program=$2

    # Leer el archivo YAML y procesar los parámetros
    while IFS= read -r line; do
        # Ejecutar el programa con los parámetros
        python "$program" $line
    done < <(python -c "import yaml; print(yaml.safe_load(open('$yaml_file')))" | \
             jq -r '["program", "method", "parameters"] as $keys | 
                    ($keys, recurse | objects | to_entries | 
                     map([.key, .value]) | select(length == 2 and all(.[]; type == "array")) | 
                     [.[].value] | combine | keys_unsorted as $params | 
                     [$keys + $params] | join("\t")) as $header | 
                     map(.[$header]) | transpose[] as $row | 
                     $header + "\t" + ($row | map(tostring) | join("\t"))' | \
             while IFS=$'\t' read -r -a fields; do
                # Generar todas las combinaciones de parámetros
                generate_combinations "${fields[0]}" "${fields[@]:1}"
             done)
}

# Verificar la cantidad de argumentos
if [ $# -ne 2 ]; then
    echo "Usage: $0 <yaml_file> <program>"
    exit 1
fi

# Asignar los argumentos a variables
yaml_file=$1
program=$2

# Verificar si el archivo YAML existe
if [ ! -f "$yaml_file" ]; then
    echo "Error: YAML file '$yaml_file' not found."
    exit 1
fi

# Verificar si el programa Python existe
if [ ! -f "$program" ]; then
    echo "Error: Python program '$program' not found."
    exit 1
fi

# Ejecutar el script Python con las combinaciones de parámetros
execute_python "$yaml_file" "$program"
