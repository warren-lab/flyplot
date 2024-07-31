# This will run the bash script tools/install.sh
import subprocess
def run():
    print("Running Installation")
    subprocess.run(['install.sh'],check=True)
# if __name__ == "__main__":
#     main()