{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      # TODO: other systems
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.rocmSupport = true;
      };
    in
    {
      devShell.${system} = pkgs.mkShell {
        buildInputs =
          [
            (pkgs.python3.withPackages (p: [
              p.torch
              p.tqdm
              p.pyvista
              p.pip
              p.matplotlib
              p.datasets
            ]))
          ];

        shellHook = ''
          echo
          ${pkgs.figlet}/bin/figlet "gptvis"

          echo "$ python --version"
          python --version
          echo
          echo "$ pip list"
          pip list
        '';
      };
    };
}
