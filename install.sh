#!/usr/bin/env bash
#
# optimisation_solver - End-User Installation Script
# Smart India Hackathon (SIH 2026) | Problem ID: SIH26119
# Mangalore Refinery and Petrochemicals Limited (MRPL)
#
# Usage:
#   ./install.sh
#

set -euo pipefail

# 1. Determine repository root reliably
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$SCRIPT_DIR"

echo "=================================================="
echo "    OPTIMISATION SOLVER — INSTALLATION"
echo "    Indigenous GPU-Accelerated Optimization Solver"
echo "    SIH26119 • MRPL"
echo "=================================================="
echo ""

# 2. Dependency checks
echo "==> Checking prerequisites..."

if ! command -v cmake >/dev/null 2>&1; then
    echo "Error: 'cmake' is required but not found in PATH." >&2
    echo "Please install CMake (3.16 or higher):" >&2
    echo "  - macOS: brew install cmake  (or install from https://cmake.org)" >&2
    echo "  - Linux: sudo apt install cmake  (or distro package manager)" >&2
    exit 1
fi

CMAKE_VERSION="$(cmake --version | head -n1)"
echo "    Found: $CMAKE_VERSION"

CXX_COMPILER=""
if [ -n "${CXX:-}" ]; then
    if command -v "$CXX" >/dev/null 2>&1; then
        CXX_COMPILER="$CXX"
    else
        echo "Error: Specified CXX compiler '$CXX' not found in PATH." >&2
        exit 1
    fi
else
    for cand in clang++ g++ c++; do
        if command -v "$cand" >/dev/null 2>&1; then
            CXX_COMPILER="$cand"
            break
        fi
    done
fi

if [ -z "$CXX_COMPILER" ]; then
    echo "Error: No suitable C++ compiler (clang++, g++, c++) found in PATH." >&2
    echo "Please install a C++17 compiler:" >&2
    echo "  - macOS: xcode-select --install" >&2
    echo "  - Linux: sudo apt install build-essential" >&2
    exit 1
fi

echo "    Found C++ compiler: $(command -v "$CXX_COMPILER")"
export CXX="$CXX_COMPILER"

# 3. Setup installation paths
INSTALL_PREFIX="${OPTIMSOLVER_PREFIX:-$HOME/.local}"
BUILD_DIR="${OPTIMSOLVER_BUILD_DIR:-$REPO_ROOT/build}"
BIN_DIR="$INSTALL_PREFIX/bin"
BIN_PATH="$BIN_DIR/optimsolver"

echo ""
echo "==> Build directory:        $BUILD_DIR"
echo "==> Installation directory: $INSTALL_PREFIX"
echo ""

# 4. Configure build
echo "==> Configuring CMake project..."
cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

# 5. Build project
echo ""
echo "==> Building optimsolver..."
cmake --build "$BUILD_DIR" -j

# 6. Install project
echo ""
echo "==> Installing executable..."
cmake --install "$BUILD_DIR" --prefix "$INSTALL_PREFIX"

if [ ! -f "$BIN_PATH" ]; then
    echo "Error: Expected executable not found at '$BIN_PATH' after install." >&2
    exit 1
fi

chmod +x "$BIN_PATH"

# 7. Check and configure PATH
echo ""
echo "==> Checking PATH configuration..."

path_in_env=0
case ":$PATH:" in
    *":$BIN_DIR:"*) path_in_env=1 ;;
esac

# Also check for '~/.local/bin' if INSTALL_PREFIX is $HOME/.local
if [ "$INSTALL_PREFIX" = "$HOME/.local" ]; then
    case ":$PATH:" in
        *":$HOME/.local/bin:"*) path_in_env=1 ;;
    esac
fi

# Detect user shell config file
USER_SHELL="$(basename "${SHELL:-zsh}")"
CONFIG_FILE=""
if [ "$USER_SHELL" = "zsh" ]; then
    CONFIG_FILE="$HOME/.zshrc"
elif [ "$USER_SHELL" = "bash" ]; then
    if [ "$(uname -s)" = "Darwin" ]; then
        if [ -f "$HOME/.bash_profile" ]; then
            CONFIG_FILE="$HOME/.bash_profile"
        elif [ -f "$HOME/.bashrc" ]; then
            CONFIG_FILE="$HOME/.bashrc"
        else
            CONFIG_FILE="$HOME/.bash_profile"
        fi
    else
        if [ -f "$HOME/.bashrc" ]; then
            CONFIG_FILE="$HOME/.bashrc"
        elif [ -f "$HOME/.bash_profile" ]; then
            CONFIG_FILE="$HOME/.bash_profile"
        else
            CONFIG_FILE="$HOME/.bashrc"
        fi
    fi
else
    # Unknown/other shell fallback
    if [ -f "$HOME/.zshrc" ]; then
        CONFIG_FILE="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        CONFIG_FILE="$HOME/.bashrc"
    fi
fi

configured_in_rc=0
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    if grep -qF "$BIN_DIR" "$CONFIG_FILE" 2>/dev/null || grep -q '\.local/bin' "$CONFIG_FILE" 2>/dev/null; then
        configured_in_rc=1
    fi
fi

updated_rc=0
if [ "$path_in_env" -eq 0 ]; then
    if [ "$configured_in_rc" -eq 0 ] && [ -n "$CONFIG_FILE" ]; then
        # Append without overwriting
        {
            echo ""
            echo "# Added by optimisation_solver installer"
            echo "export PATH=\"$BIN_DIR:\$PATH\""
        } >> "$CONFIG_FILE"
        updated_rc=1
    fi
fi

echo ""
echo "=================================================="
echo "    INSTALLATION SUCCESSFUL!"
echo "=================================================="
echo "  Installed binary: $BIN_PATH"
echo ""

if [ "$path_in_env" -eq 1 ]; then
    echo "  [PATH Status] '$BIN_DIR' is already in your PATH."
    echo "  You can immediately run:"
    echo "      optimsolver"
elif [ "$updated_rc" -eq 1 ]; then
    echo "  [PATH Status] Added '$BIN_DIR' to $CONFIG_FILE."
    echo ""
    echo "  IMPORTANT: A child installer script cannot modify your current shell."
    echo "  To use 'optimsolver' in this terminal session, run:"
    echo "      source $CONFIG_FILE"
    echo "  or open a new terminal tab/window."
elif [ "$configured_in_rc" -eq 1 ]; then
    echo "  [PATH Status] '$BIN_DIR' is configured in $CONFIG_FILE, but not yet loaded"
    echo "  in this active shell session."
    echo "  To use 'optimsolver' in this terminal session, run:"
    echo "      source $CONFIG_FILE"
    echo "  or open a new terminal tab/window."
else
    echo "  [PATH Status] '$BIN_DIR' is NOT currently in your PATH."
    echo "  Add it manually by running:"
    echo "      export PATH=\"$BIN_DIR:\$PATH\""
fi

echo ""
echo "  Usage:"
echo "    optimsolver                     # Launch interactive menu"
echo "    optimsolver solve <model.mps>   # Non-interactive batch solve"
echo "=================================================="
echo ""
