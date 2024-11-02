class Reward:
    @classmethod
    def _calculate_rewards(cls, action1, action2):
        """Calculate rewards based on actions."""
        if action1 == action2:
            return 1, 1
        return -1, -1

    @classmethod
    def _calculate_rewards_norm_change_(cls, action1, action2, less_action):
        if action1 == less_action:
            reward1 = 0.5
        else:
            reward1 = -0.5

        if action2 == less_action:
            reward2 = 0.5
        else:
            reward2 = -0.5

        return reward1, reward2
