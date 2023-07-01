import sys

def add():
    
    var1 = float(sys.argv[1])
    var2 = float(sys.argv[2])

    with open("file.txt", "w") as file:
        file.write(f'{var1 + var2}')
        file.write(f'{var1 - var2}')

    return 0

if __name__ == "__main__":
    add()