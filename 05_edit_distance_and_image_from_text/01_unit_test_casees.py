from min_max import min_max_no
import unittest

class TestSum(unittest.TestCase):
    def test_case_1(self):
        """
        Normal test case with numeric digit
        """
        data = [1, 2, 3, 4, 5, 6, 7]
        maxNo, minNo = min_max_no(data)
        self.assertEqual(maxNo == 7 and minNo == 1, True)

    def test_case_2(self):
        """
        To test if we pass char in numeric array
        """
        data = [1, 2, 3, 4, 5, 6, 'a']
        maxNo, minNo = min_max_no(data)
        self.assertEqual(maxNo == 0 and minNo == 0, True)

    def test_case_3(self):
        """
        With duplicates
        """
        data = [99,99]
        maxNo, minNo = min_max_no(data)
        self.assertEqual(maxNo == 99 and minNo == 99, True)

if __name__ == '__main__':
    unittest.main()