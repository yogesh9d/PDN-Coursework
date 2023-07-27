import socket

def create_udp_server():
    udp_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_server.bind(("127.0.0.1", 16278))
    return udp_server

def main():
    server_socket = create_udp_server()
    print("Listening on 127.0.0.1:16278")

    while True:
        received_data, sender_address = server_socket.recvfrom(1024)
        input_numbers = received_data.decode()

        try:
            num_list = [float(num) for num in input_numbers.split(",")]
            total_product = 1
            for num in num_list:
                total_product *= num
            server_socket.sendto(str(total_product).encode(), sender_address)
        except:
            server_socket.sendto("Invalid input".encode(), sender_address)

if __name__ == "__main__":
    main()
