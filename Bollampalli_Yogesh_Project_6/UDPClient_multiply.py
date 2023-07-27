import socket

def generate_udp_socket():
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def main():
    udp_socket = generate_udp_socket()

    input_numbers = input("Enter comma-separated numbers: ")
    udp_socket.sendto(input_numbers.encode(), ("127.0.0.1", 16278))

    received_data, _ = udp_socket.recvfrom(1024)
    print("Received:", received_data.decode())

    udp_socket.close()

if __name__ == "__main__":
    main()
