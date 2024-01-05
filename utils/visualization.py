# Value 시각화 함수 만들기
def value_visualize(episode, data):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.imshow(data, interpolation='nearest')
    plt.colorbar()

    # 축 눈금을 1부터 4로 설정
    plt.xticks(range(4), [0, 1, 2, 3])
    plt.yticks(range(4), [0, 1, 2, 3])

    for i in range(4):
        for j in range(4):
            plt.text(j, i, '{:.2f}'.format(data[i][j]),
                    ha='center', va='center', color='white')
    plt.title(f"Episode: {episode + 1}")
    plt.show()

# policy 시각화 함수 만들기
def action_visualize(data):
    import matplotlib.pyplot as plt
    plt.imshow(data, interpolation='nearest', cmap=plt.get_cmap('Paired'))
    plt.xticks(range(7), [0, 1, 2, 3, 4, 5, 6])
    plt.yticks(range(5), [0, 1, 2, 3, 4])

    d_symbols = ['←','↑', '→', '↓', '']
    for i in range(5):
        for j in range(7):
            if (j, i) in [(2, 0), (2, 1), (2, 2), (4, 2), (4, 3), (4, 4)]:
                direction = d_symbols[-1]
            elif (j, i) == (6, 4):
                direction = 'G'
            else:
                direction = int(data[i,j])
                direction = d_symbols[direction]
            plt.text(j, i, direction, ha="center", va="center", color="black", fontsize=20)
    plt.title(f"Optimal Policy")
    plt.show()