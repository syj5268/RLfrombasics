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