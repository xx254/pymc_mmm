import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('data/mmm_example.csv')
data['date_week'] = pd.to_datetime(data['date_week'])

# 创建图表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# 上半部分：销售额和事件
ax1.plot(data['date_week'], data['y'], color='gray', alpha=0.5, label='Sales')
# 标记事件1
event1_mask = data['event_1'] == 1
ax1.scatter(data[event1_mask]['date_week'], data[event1_mask]['y'], 
           color='blue', s=100, label='Event 1')
# 标记事件2
event2_mask = data['event_2'] == 1
ax1.scatter(data[event2_mask]['date_week'], data[event2_mask]['y'], 
           color='red', s=100, label='Event 2')

ax1.set_title('Sales and Events Timeline')
ax1.set_ylabel('Sales')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 下半部分：营销投入
ax2.plot(data['date_week'], data['x1'], color='blue', alpha=0.5, label='Social Media (x1)')
ax2.plot(data['date_week'], data['x2'], color='red', alpha=0.5, label='Search Engine (x2)')
# 标记事件发生时的营销投入
ax2.scatter(data[event1_mask]['date_week'], data[event1_mask]['x1'], 
           color='blue', s=100)
ax2.scatter(data[event1_mask]['date_week'], data[event1_mask]['x2'], 
           color='blue', s=100)
ax2.scatter(data[event2_mask]['date_week'], data[event2_mask]['x1'], 
           color='red', s=100)
ax2.scatter(data[event2_mask]['date_week'], data[event2_mask]['x2'], 
           color='red', s=100)

ax2.set_title('Marketing Channels Intensity')
ax2.set_xlabel('Date')
ax2.set_ylabel('Marketing Intensity')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('plots/events_visualization.png')

# 打印事件发生时的具体信息
print("\nEvent 1 Details:")
print(data[event1_mask][['date_week', 'y', 'x1', 'x2']].to_string())
print("\nEvent 2 Details:")
print(data[event2_mask][['date_week', 'y', 'x1', 'x2']].to_string()) 