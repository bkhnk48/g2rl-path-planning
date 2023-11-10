import pickle  #Nhập mô-đun pickle để lưu và tải các đối tượng Python như bảng Q.
def training():  #Định nghĩa một hàm để huấn luyện bảng Q trên một môi trường kho hàng.
    import random
    import numpy as np
    from environment import WarehouseEnvironment

    env = WarehouseEnvironment()    # Tạo một đối tượng env là một phiên bản của lớp WarehouseEnvironment.
    q_table = np.zeros([env.n_states, env.n_actions]) # Tạo một bảng Q là một mảng numpy có kích thước bằng số trạng thái và số hành động của môi trường, và khởi tạo tất cả các giá trị bằng 0.
    # Hyperparameters
    alpha = 0.3  # Đặt hệ số học alpha bằng 0.3, đây là một số thực trong khoảng từ 0 đến 1 để xác định mức độ cập nhật giá trị Q dựa trên kinh nghiệm mới.
    gamma = 0.9  # Đặt hệ số giảm gamma bằng 0.9, đây là một số thực trong khoảng từ 0 đến 1 để xác định mức độ ảnh hưởng của các phần thưởng tương lai đến giá trị Q hiện tại.
    epsilon = 0.1  # Đặt tham số thăm dò epsilon bằng 0.1, đây là một số thực trong khoảng từ 0 đến 1 để xác định xác suất của việc chọn một hành động ngẫu nhiên thay vì hành động tối ưu theo bảng Q.

    # For plotting metrics  # Đây là một dòng nhận xét để chỉ ra rằng các biến sau đây là để lưu trữ các số liệu đánh giá quá trình huấn luyện.
    rewards_window = []  # Tạo một danh sách trống để lưu trữ giá trị trung bình của phần thưởng trong các lần lặp cuối cùng.
    all_rewards = []     # Tạo một danh sách trống để lưu trữ giá trị phần thưởng của mỗi lần lặp.

    for i in range(1, 10001):  # chạy 10,000 lần lặp huấn luyện
        state,_ = env.reset()  # Đặt lại môi trường và nhận trạng thái ban đầu của nó, bỏ qua giá trị phần thưởng trả về bởi phương thức reset.

        epochs, penalties, reward, = 0, 0, 0 # epochs chỉ ra số lần agent thực hiện các hành động và nhận được phần thưởng trong môi trường. Mỗi epoch có thể bao gồm nhiều bước (steps), tùy thuộc vào chiều dài của mỗi tập (episode
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.action_space()) # Explore action space # Chọn một hành động ngẫu nhiên từ không gian hành động của môi trường
            else:
                action = np.argmax(q_table[state]) # Exploit learned values  # Chọn hành động có giá trị Q cao nhất cho trạng thái hiện tại, đây là cách khai thác các giá trị đã học.

            _, next_state, reward, done = env.step(action) # Thực hiện hành động đã chọn trên môi trường và nhận trạng thái tiếp theo, phần thưởng, và biến done để biết lần lặp có kết thúc hay không, bỏ qua giá trị quan sát trả về bởi phương thức step.
            
            old_value = q_table[state, action]  # Lấy giá trị Q cũ cho cặp trạng thái và hành động hiện tại từ bảng Q.
            next_max = np.max(q_table[next_state]) # Tìm giá trị Q lớn nhất cho trạng thái tiếp theo từ bảng Q, đây là giá trị Q tối ưu cho trạng thái đó.
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max) # Tính toán giá trị Q mới cho cặp trạng thái và hành động hiện tại theo công thức Q-learning, đây là cách cập nhật giá trị Q dựa trên kinh nghiệm mới và các phần thưởng tương lai.
            q_table[state, action] = new_value # Gán giá trị Q mới vào bảng Q cho cặp trạng thái và hành động hiện tại, đây là cách học giá trị Q tốt nhất cho mỗi cặp trạng thái và hành động.

            if reward <= -0.1:
                penalties += 1 # Tăng biến penalties lên 1, để đếm số lần bị phạt trong một lần lặp.
            
            state = next_state
            epochs += 1    # Tăng biến epochs lên 1, để đếm số bước trong một lần lặp.

        env.create_scenes("data/agents_q_learning.gif")
        all_rewards.append(reward)

        if i % 100 == 0:
            # env.create_scenes(f"data/agents_locals_q_learning_{i}.gif")
            rewards_window.append(sum(all_rewards[-100:])/100) # Dòng này thêm giá trị trung bình của phần thưởng (reward) trong 100 episode gần nhất vào danh sách rewards_window. Phần thưởng là một số thực để đánh giá hiệu quả của hành động của tác tử. Danh sách rewards_window dùng để theo dõi sự cải thiện của tác tử qua các episode.    
            print(f"Episode: {i} reward: {reward}")   # Đây là một cách để kiểm tra tiến trình của quá trình huấn luyện.

    print("Training finished.\n")

    return q_table, rewards_window, all_rewards

q_tbl, r_win, all_r = training()

with open('./models/q_learning_table.pkl','wb') as f: # Dòng này mở một file có tên là q_learning_table.pkl ở chế độ ghi nhị phân (wb) và gán đối tượng file vào biến f. File này nằm trong thư mục models của đường dẫn hiện tại. Dòng này cũng sử dụng cấu trúc with để đảm bảo file được đóng lại sau khi thực hiện các lệnh bên trong khối with.
    pickle.dump(q_tbl, f) # lưu biến q_tbl vào file f.

with open('./models/rewards_window.pkl','wb') as f: mở một file có tên là rewards_window.pkl và gán đối tượng file vào biến f
    pickle.dump(r_win, f) # Dòng này lưu biến r_win vào file

with open('./models/all_rewards.pkl','wb') as f:  # mở file có tên là all_rewards.pkl
    pickle.dump(all_r, f)  # lưu biến all_r vào file
