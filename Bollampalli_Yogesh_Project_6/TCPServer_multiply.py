import socket

def initialize_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", 16278))
    server_socket.listen(1)
    return server_socket

def main():
    tcp_server = initialize_server()
    print("Listening on 127.0.0.1:16278")

    while True:
        client_socket, client_addr = tcp_server.accept()
        input_data = client_socket.recv(1024).decode()

        try:
            number_list = [float(num) for num in input_data.split(",")]
            total_product = 1
            for num in number_list:
                total_product *= num
            client_socket.sendall(str(total_product).encode())
        except:
            client_socket.sendall("Invalid input".encode())

        client_socket.close()

if __name__ == "__main__":
    main()
