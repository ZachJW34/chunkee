# Chunkee

Chunkee is a voxel engine written in Rust, designed to be integrated with the Godot game engine.

## Project Structure

The project is structured as a Cargo workspace and is made up of two main components:

-   `chunkee-core`: The heart of the engine, responsible for core functionalities like world generation, chunk management, and meshing.
-   `chunkee-godot`: A GDExtension that bridges the `chunkee-core` library with Godot, allowing you to use the engine's features within a Godot project.