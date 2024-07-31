# This will run the bash script tools/install.sh
import subprocess
import importlib.resources as pkg_resources
def run():
    print("Running Installation")
    # Determine the directory of the current script
    with pkg_resources.path('tools', 'install.sh') as script_dir:
        print(script_dir)
    # Run subprocess
        subprocess.run([script_dir],check=True)
# if __name__ == "__main__":
#     main()