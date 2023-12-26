# Value 시각화 함수 만들기
def value_visualize(episode, data):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.imshow(data, interpolation='nearest')
    plt.colorbar()
    for i in range(4):
        for j in range(4):
            plt.text(j, i, '{:.2f}'.format(data[i][j]),
                    ha='center', va='center', color='white')
    plt.title(f"Episode: {episode + 1}")
    plt.show()