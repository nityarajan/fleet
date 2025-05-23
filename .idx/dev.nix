{ pkgs, ... }: {
  channel = "stable-24.05";

  packages = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.venvShellHook
  ];

  idx = {
    extensions = [ "ms-python.python" ];
    workspace = {
      onCreate = {
        install = ''
          python -m venv .venv
          source .venv/bin/activate
          pip install --upgrade pip
          pip install -r requirements.txt
        '';
        default.openFiles = [ "README.md" "src/index.html" "main.py" ];
      };
    };

    previews = {
      enable = true;
      previews = {
        web = {
          command = [ "python" "main.py" ];
          env = { PORT = "$PORT"; };
          manager = "web";
        };
      };
    };
  };
}
