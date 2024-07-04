import unittest
import numpy as np

class TestProcessGrippers(unittest.TestCase):
    def test_process_grippers(self):
        grippers = [{
            'tcp_pose': np.array([[1, 2, 3, 4, 5, 6]]),
            'gripper_width': [0.5],
            'demo_start_pose': np.array([[7, 8, 9, 10, 11, 12]]),
            'demo_end_pose': np.array([[13, 14, 15, 16, 17, 18]])
        }]
        
        result = process_grippers(grippers)
        self.assertEqual(result['eef_pos'].shape, (1, 3))
        self.assertEqual(result['eef_rot_axis_angle'].shape, (1, 3))
        self.assertEqual(result['gripper_width'].shape, (1, 1))
        self.assertEqual(result['demo_start_pose'].shape, (1, 6))
        self.assertEqual(result['demo_end_pose'].shape, (1, 6))

if __name__ == '__main__':
    unittest.main()
