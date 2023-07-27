import socket

def create_tcp_connection():
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.connect(("127.0.0.1", 16278))
    return connection

def main():
    tcp_conn = create_tcp_connection()

    input_numbers = input("Enter comma-separated numbers: ")
    tcp_conn.sendall(input_numbers.encode())

    response = tcp_conn.recv(1024).decode()
    print("Received:", response)

    tcp_conn.close()

if __name__ == "__main__":
    main()
