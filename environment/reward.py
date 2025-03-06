class Reward:
    @classmethod
    def calculate_rewards(cls, action1, action2):
        """Calculate rewards based on actions."""
        if action1 == action2:
            return 1, 1
        return -1, -1