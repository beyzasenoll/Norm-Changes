class Reward:
    @classmethod
    def calculate_rewards(cls, action1, action2):
        """Calculate rewards based on actions."""
        if action1 == action2:
            return 1, 1
        return -1, -1

    @classmethod
    def calculate_rewards_norm_change_(cls, action1, action2, less_action, reward=0.5):
        if action1 == action2:
            if action1 != less_action:
                return reward, reward
            else:
                return 1,1
        return -1, -1