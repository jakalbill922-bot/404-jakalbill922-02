import numpy as np

# https://github.com/andrewwillmott/sh-lib

kSqrt02_01 = np.sqrt(2.0 / 1.0)
kSqrt01_02 = np.sqrt(1.0 / 2.0)
kSqrt03_02 = np.sqrt(3.0 / 2.0)
kSqrt01_03 = np.sqrt(1.0 / 3.0)
kSqrt02_03 = np.sqrt(2.0 / 3.0)
kSqrt04_03 = np.sqrt(4.0 / 3.0)
kSqrt01_04 = np.sqrt(1.0 / 4.0)
kSqrt03_04 = np.sqrt(3.0 / 4.0)
kSqrt05_04 = np.sqrt(5.0 / 4.0)
kSqrt01_05 = np.sqrt(1.0 / 5.0)
kSqrt02_05 = np.sqrt(2.0 / 5.0)
kSqrt03_05 = np.sqrt(3.0 / 5.0)
kSqrt04_05 = np.sqrt(4.0 / 5.0)
kSqrt06_05 = np.sqrt(6.0 / 5.0)
kSqrt08_05 = np.sqrt(8.0 / 5.0)
kSqrt09_05 = np.sqrt(9.0 / 5.0)
kSqrt01_06 = np.sqrt(1.0 / 6.0)
kSqrt05_06 = np.sqrt(5.0 / 6.0)
kSqrt07_06 = np.sqrt(7.0 / 6.0)
kSqrt02_07 = np.sqrt(02.0 / 7.0)
kSqrt06_07 = np.sqrt(6.0 / 7.0)
kSqrt10_07 = np.sqrt(10.0 / 7.0)
kSqrt12_07 = np.sqrt(12.0 / 7.0)
kSqrt15_07 = np.sqrt(15.0 / 7.0)
kSqrt16_07 = np.sqrt(16.0 / 7.0)
kSqrt01_08 = np.sqrt(1.0 / 8.0)
kSqrt03_08 = np.sqrt(3.0 / 8.0)
kSqrt05_08 = np.sqrt(5.0 / 8.0)
kSqrt07_08 = np.sqrt(7.0 / 8.0)
kSqrt09_08 = np.sqrt(9.0 / 8.0)
kSqrt05_09 = np.sqrt(5.0 / 9.0)
kSqrt08_09 = np.sqrt(8.0 / 9.0)
kSqrt01_10 = np.sqrt(1.0 / 10.0)
kSqrt03_10 = np.sqrt(3.0 / 10.0)
kSqrt07_10 = np.sqrt(7.0 / 10.0)
kSqrt09_10 = np.sqrt(9.0 / 10.0)
kSqrt01_12 = np.sqrt(1.0 / 12.0)
kSqrt07_12 = np.sqrt(7.0 / 12.0)
kSqrt11_12 = np.sqrt(11.0 / 12.0)
kSqrt01_14 = np.sqrt(1.0 / 14.0)
kSqrt03_14 = np.sqrt(3.0 / 14.0)
kSqrt15_14 = np.sqrt(15.0 / 14.0)
kSqrt04_15 = np.sqrt(4.0 / 15.0)
kSqrt07_15 = np.sqrt(7.0 / 10.0)
kSqrt14_15 = np.sqrt(14.0 / 15.0)
kSqrt16_15 = np.sqrt(16.0 / 15.0)
kSqrt01_16 = np.sqrt(1.0 / 16.0)
kSqrt03_16 = np.sqrt(3.0 / 16.0)
kSqrt07_16 = np.sqrt(7.0 / 16.0)
kSqrt15_16 = np.sqrt(15.0 / 16.0)
kSqrt01_18 = np.sqrt(1.0 / 18.0)
kSqrt01_24 = np.sqrt(1.0 / 24.0)
kSqrt03_25 = np.sqrt(3.0 / 25.0)
kSqrt09_25 = np.sqrt(9.0 / 25.0)
kSqrt14_25 = np.sqrt(14.0 / 25.0)
kSqrt16_25 = np.sqrt(16.0 / 25.0)
kSqrt18_25 = np.sqrt(18.0 / 25.0)
kSqrt21_25 = np.sqrt(21.0 / 25.0)
kSqrt24_25 = np.sqrt(24.0 / 25.0)
kSqrt03_28 = np.sqrt(3.0 / 28.0)
kSqrt05_28 = np.sqrt(5.0 / 28.0)
kSqrt01_30 = np.sqrt(1.0 / 30.0)
kSqrt01_32 = np.sqrt(1.0 / 32.0)
kSqrt03_32 = np.sqrt(3.0 / 32.0)
kSqrt15_32 = np.sqrt(15.0 / 32.0)
kSqrt21_32 = np.sqrt(21.0 / 32.0)
kSqrt11_36 = np.sqrt(11.0 / 36.0)
kSqrt35_36 = np.sqrt(35.0 / 36.0)
kSqrt01_50 = np.sqrt(1.0 / 50.0)
kSqrt03_50 = np.sqrt(3.0 / 50.0)
kSqrt21_50 = np.sqrt(21.0 / 50.0)
kSqrt15_56 = np.sqrt(15.0 / 56.0)
kSqrt01_60 = np.sqrt(1.0 / 60.0)
kSqrt01_112 = np.sqrt(1.0 / 112.0)
kSqrt03_112 = np.sqrt(3.0 / 112.0)
kSqrt15_112 = np.sqrt(15.0 / 112.0)


def get_sh1(R):
    return np.roll(np.roll(R, -1, axis=-1), -1, axis=-2)


def get_sh2(sh1):
    sh2 = np.zeros((5, 5), dtype=sh1.dtype)

    sh2[0][0] = kSqrt01_04 * (
                (sh1[2][2] * sh1[0][0] + sh1[2][0] * sh1[0][2]) + (sh1[0][2] * sh1[2][0] + sh1[0][0] * sh1[2][2]))
    sh2[0][1] = (sh1[2][1] * sh1[0][0] + sh1[0][1] * sh1[2][0])
    sh2[0][2] = kSqrt03_04 * (sh1[2][1] * sh1[0][1] + sh1[0][1] * sh1[2][1])
    sh2[0][3] = (sh1[2][1] * sh1[0][2] + sh1[0][1] * sh1[2][2])
    sh2[0][4] = kSqrt01_04 * (
                (sh1[2][2] * sh1[0][2] - sh1[2][0] * sh1[0][0]) + (sh1[0][2] * sh1[2][2] - sh1[0][0] * sh1[2][0]))

    sh2[1][0] = kSqrt01_04 * (
                (sh1[1][2] * sh1[0][0] + sh1[1][0] * sh1[0][2]) + (sh1[0][2] * sh1[1][0] + sh1[0][0] * sh1[1][2]))
    sh2[1][1] = sh1[1][1] * sh1[0][0] + sh1[0][1] * sh1[1][0]
    sh2[1][2] = kSqrt03_04 * (sh1[1][1] * sh1[0][1] + sh1[0][1] * sh1[1][1])
    sh2[1][3] = sh1[1][1] * sh1[0][2] + sh1[0][1] * sh1[1][2]
    sh2[1][4] = kSqrt01_04 * (
                (sh1[1][2] * sh1[0][2] - sh1[1][0] * sh1[0][0]) + (sh1[0][2] * sh1[1][2] - sh1[0][0] * sh1[1][0]))

    sh2[2][0] = kSqrt01_03 * (sh1[1][2] * sh1[1][0] + sh1[1][0] * sh1[1][2]) - kSqrt01_12 * (
                (sh1[2][2] * sh1[2][0] + sh1[2][0] * sh1[2][2]) + (sh1[0][2] * sh1[0][0] + sh1[0][0] * sh1[0][2]))
    sh2[2][1] = kSqrt04_03 * sh1[1][1] * sh1[1][0] - kSqrt01_03 * (sh1[2][1] * sh1[2][0] + sh1[0][1] * sh1[0][0])
    sh2[2][2] = sh1[1][1] * sh1[1][1] - kSqrt01_04 * (sh1[2][1] * sh1[2][1] + sh1[0][1] * sh1[0][1])
    sh2[2][3] = kSqrt04_03 * sh1[1][1] * sh1[1][2] - kSqrt01_03 * (sh1[2][1] * sh1[2][2] + sh1[0][1] * sh1[0][2])
    sh2[2][4] = kSqrt01_03 * (sh1[1][2] * sh1[1][2] - sh1[1][0] * sh1[1][0]) - kSqrt01_12 * (
                (sh1[2][2] * sh1[2][2] - sh1[2][0] * sh1[2][0]) + (sh1[0][2] * sh1[0][2] - sh1[0][0] * sh1[0][0]))

    sh2[3][0] = kSqrt01_04 * (
                (sh1[1][2] * sh1[2][0] + sh1[1][0] * sh1[2][2]) + (sh1[2][2] * sh1[1][0] + sh1[2][0] * sh1[1][2]))
    sh2[3][1] = sh1[1][1] * sh1[2][0] + sh1[2][1] * sh1[1][0]
    sh2[3][2] = kSqrt03_04 * (sh1[1][1] * sh1[2][1] + sh1[2][1] * sh1[1][1])
    sh2[3][3] = sh1[1][1] * sh1[2][2] + sh1[2][1] * sh1[1][2]
    sh2[3][4] = kSqrt01_04 * (
                (sh1[1][2] * sh1[2][2] - sh1[1][0] * sh1[2][0]) + (sh1[2][2] * sh1[1][2] - sh1[2][0] * sh1[1][0]))

    sh2[4][0] = kSqrt01_04 * (
                (sh1[2][2] * sh1[2][0] + sh1[2][0] * sh1[2][2]) - (sh1[0][2] * sh1[0][0] + sh1[0][0] * sh1[0][2]))
    sh2[4][1] = (sh1[2][1] * sh1[2][0] - sh1[0][1] * sh1[0][0])
    sh2[4][2] = kSqrt03_04 * (sh1[2][1] * sh1[2][1] - sh1[0][1] * sh1[0][1])
    sh2[4][3] = (sh1[2][1] * sh1[2][2] - sh1[0][1] * sh1[0][2])
    sh2[4][4] = kSqrt01_04 * (
                (sh1[2][2] * sh1[2][2] - sh1[2][0] * sh1[2][0]) - (sh1[0][2] * sh1[0][2] - sh1[0][0] * sh1[0][0]))

    return sh2


def get_sh3(sh1, sh2):
    sh3 = np.zeros((7, 7), dtype=sh1.dtype)

    sh3[0][0] = kSqrt01_04 * (
                (sh1[2][2] * sh2[0][0] + sh1[2][0] * sh2[0][4]) + (sh1[0][2] * sh2[4][0] + sh1[0][0] * sh2[4][4]))
    sh3[0][1] = kSqrt03_02 * (sh1[2][1] * sh2[0][0] + sh1[0][1] * sh2[4][0])
    sh3[0][2] = kSqrt15_16 * (sh1[2][1] * sh2[0][1] + sh1[0][1] * sh2[4][1])
    sh3[0][3] = kSqrt05_06 * (sh1[2][1] * sh2[0][2] + sh1[0][1] * sh2[4][2])
    sh3[0][4] = kSqrt15_16 * (sh1[2][1] * sh2[0][3] + sh1[0][1] * sh2[4][3])
    sh3[0][5] = kSqrt03_02 * (sh1[2][1] * sh2[0][4] + sh1[0][1] * sh2[4][4])
    sh3[0][6] = kSqrt01_04 * (
                (sh1[2][2] * sh2[0][4] - sh1[2][0] * sh2[0][0]) + (sh1[0][2] * sh2[4][4] - sh1[0][0] * sh2[4][0]))

    sh3[1][0] = kSqrt01_06 * (sh1[1][2] * sh2[0][0] + sh1[1][0] * sh2[0][4]) + kSqrt01_06 * (
                (sh1[2][2] * sh2[1][0] + sh1[2][0] * sh2[1][4]) + (sh1[0][2] * sh2[3][0] + sh1[0][0] * sh2[3][4]))
    sh3[1][1] = sh1[1][1] * sh2[0][0] + (sh1[2][1] * sh2[1][0] + sh1[0][1] * sh2[3][0])
    sh3[1][2] = kSqrt05_08 * sh1[1][1] * sh2[0][1] + kSqrt05_08 * (sh1[2][1] * sh2[1][1] + sh1[0][1] * sh2[3][1])
    sh3[1][3] = kSqrt05_09 * sh1[1][1] * sh2[0][2] + kSqrt05_09 * (sh1[2][1] * sh2[1][2] + sh1[0][1] * sh2[3][2])
    sh3[1][4] = kSqrt05_08 * sh1[1][1] * sh2[0][3] + kSqrt05_08 * (sh1[2][1] * sh2[1][3] + sh1[0][1] * sh2[3][3])
    sh3[1][5] = sh1[1][1] * sh2[0][4] + (sh1[2][1] * sh2[1][4] + sh1[0][1] * sh2[3][4])
    sh3[1][6] = kSqrt01_06 * (sh1[1][2] * sh2[0][4] - sh1[1][0] * sh2[0][0]) + kSqrt01_06 * (
                (sh1[2][2] * sh2[1][4] - sh1[2][0] * sh2[1][0]) + (sh1[0][2] * sh2[3][4] - sh1[0][0] * sh2[3][0]))

    sh3[2][0] = kSqrt04_15 * (sh1[1][2] * sh2[1][0] + sh1[1][0] * sh2[1][4]) + kSqrt01_05 * (
                sh1[0][2] * sh2[2][0] + sh1[0][0] * sh2[2][4]) - kSqrt01_60 * (
                            (sh1[2][2] * sh2[0][0] + sh1[2][0] * sh2[0][4]) - (
                                sh1[0][2] * sh2[4][0] + sh1[0][0] * sh2[4][4]))
    sh3[2][1] = kSqrt08_05 * sh1[1][1] * sh2[1][0] + kSqrt06_05 * sh1[0][1] * sh2[2][0] - kSqrt01_10 * (
                sh1[2][1] * sh2[0][0] - sh1[0][1] * sh2[4][0])
    sh3[2][2] = sh1[1][1] * sh2[1][1] + kSqrt03_04 * sh1[0][1] * sh2[2][1] - kSqrt01_16 * (
                sh1[2][1] * sh2[0][1] - sh1[0][1] * sh2[4][1])
    sh3[2][3] = kSqrt08_09 * sh1[1][1] * sh2[1][2] + kSqrt02_03 * sh1[0][1] * sh2[2][2] - kSqrt01_18 * (
                sh1[2][1] * sh2[0][2] - sh1[0][1] * sh2[4][2])
    sh3[2][4] = sh1[1][1] * sh2[1][3] + kSqrt03_04 * sh1[0][1] * sh2[2][3] - kSqrt01_16 * (
                sh1[2][1] * sh2[0][3] - sh1[0][1] * sh2[4][3])
    sh3[2][5] = kSqrt08_05 * sh1[1][1] * sh2[1][4] + kSqrt06_05 * sh1[0][1] * sh2[2][4] - kSqrt01_10 * (
                sh1[2][1] * sh2[0][4] - sh1[0][1] * sh2[4][4])
    sh3[2][6] = kSqrt04_15 * (sh1[1][2] * sh2[1][4] - sh1[1][0] * sh2[1][0]) + kSqrt01_05 * (
                sh1[0][2] * sh2[2][4] - sh1[0][0] * sh2[2][0]) - kSqrt01_60 * (
                            (sh1[2][2] * sh2[0][4] - sh1[2][0] * sh2[0][0]) - (
                                sh1[0][2] * sh2[4][4] - sh1[0][0] * sh2[4][0]))

    sh3[3][0] = kSqrt03_10 * (sh1[1][2] * sh2[2][0] + sh1[1][0] * sh2[2][4]) - kSqrt01_10 * (
                (sh1[2][2] * sh2[3][0] + sh1[2][0] * sh2[3][4]) + (sh1[0][2] * sh2[1][0] + sh1[0][0] * sh2[1][4]))
    sh3[3][1] = kSqrt09_05 * sh1[1][1] * sh2[2][0] - kSqrt03_05 * (sh1[2][1] * sh2[3][0] + sh1[0][1] * sh2[1][0])
    sh3[3][2] = kSqrt09_08 * sh1[1][1] * sh2[2][1] - kSqrt03_08 * (sh1[2][1] * sh2[3][1] + sh1[0][1] * sh2[1][1])
    sh3[3][3] = sh1[1][1] * sh2[2][2] - kSqrt01_03 * (sh1[2][1] * sh2[3][2] + sh1[0][1] * sh2[1][2])
    sh3[3][4] = kSqrt09_08 * sh1[1][1] * sh2[2][3] - kSqrt03_08 * (sh1[2][1] * sh2[3][3] + sh1[0][1] * sh2[1][3])
    sh3[3][5] = kSqrt09_05 * sh1[1][1] * sh2[2][4] - kSqrt03_05 * (sh1[2][1] * sh2[3][4] + sh1[0][1] * sh2[1][4])
    sh3[3][6] = kSqrt03_10 * (sh1[1][2] * sh2[2][4] - sh1[1][0] * sh2[2][0]) - kSqrt01_10 * (
                (sh1[2][2] * sh2[3][4] - sh1[2][0] * sh2[3][0]) + (sh1[0][2] * sh2[1][4] - sh1[0][0] * sh2[1][0]))

    sh3[4][0] = kSqrt04_15 * (sh1[1][2] * sh2[3][0] + sh1[1][0] * sh2[3][4]) + kSqrt01_05 * (
                sh1[2][2] * sh2[2][0] + sh1[2][0] * sh2[2][4]) - kSqrt01_60 * (
                            (sh1[2][2] * sh2[4][0] + sh1[2][0] * sh2[4][4]) + (
                                sh1[0][2] * sh2[0][0] + sh1[0][0] * sh2[0][4]))
    sh3[4][1] = kSqrt08_05 * sh1[1][1] * sh2[3][0] + kSqrt06_05 * sh1[2][1] * sh2[2][0] - kSqrt01_10 * (
                sh1[2][1] * sh2[4][0] + sh1[0][1] * sh2[0][0])
    sh3[4][2] = sh1[1][1] * sh2[3][1] + kSqrt03_04 * sh1[2][1] * sh2[2][1] - kSqrt01_16 * (
                sh1[2][1] * sh2[4][1] + sh1[0][1] * sh2[0][1])
    sh3[4][3] = kSqrt08_09 * sh1[1][1] * sh2[3][2] + kSqrt02_03 * sh1[2][1] * sh2[2][2] - kSqrt01_18 * (
                sh1[2][1] * sh2[4][2] + sh1[0][1] * sh2[0][2])
    sh3[4][4] = sh1[1][1] * sh2[3][3] + kSqrt03_04 * sh1[2][1] * sh2[2][3] - kSqrt01_16 * (
                sh1[2][1] * sh2[4][3] + sh1[0][1] * sh2[0][3])
    sh3[4][5] = kSqrt08_05 * sh1[1][1] * sh2[3][4] + kSqrt06_05 * sh1[2][1] * sh2[2][4] - kSqrt01_10 * (
                sh1[2][1] * sh2[4][4] + sh1[0][1] * sh2[0][4])
    sh3[4][6] = kSqrt04_15 * (sh1[1][2] * sh2[3][4] - sh1[1][0] * sh2[3][0]) + kSqrt01_05 * (
                sh1[2][2] * sh2[2][4] - sh1[2][0] * sh2[2][0]) - kSqrt01_60 * (
                            (sh1[2][2] * sh2[4][4] - sh1[2][0] * sh2[4][0]) + (
                                sh1[0][2] * sh2[0][4] - sh1[0][0] * sh2[0][0]))

    sh3[5][0] = kSqrt01_06 * (sh1[1][2] * sh2[4][0] + sh1[1][0] * sh2[4][4]) + kSqrt01_06 * (
                (sh1[2][2] * sh2[3][0] + sh1[2][0] * sh2[3][4]) - (sh1[0][2] * sh2[1][0] + sh1[0][0] * sh2[1][4]))
    sh3[5][1] = sh1[1][1] * sh2[4][0] + (sh1[2][1] * sh2[3][0] - sh1[0][1] * sh2[1][0])
    sh3[5][2] = kSqrt05_08 * sh1[1][1] * sh2[4][1] + kSqrt05_08 * (sh1[2][1] * sh2[3][1] - sh1[0][1] * sh2[1][1])
    sh3[5][3] = kSqrt05_09 * sh1[1][1] * sh2[4][2] + kSqrt05_09 * (sh1[2][1] * sh2[3][2] - sh1[0][1] * sh2[1][2])
    sh3[5][4] = kSqrt05_08 * sh1[1][1] * sh2[4][3] + kSqrt05_08 * (sh1[2][1] * sh2[3][3] - sh1[0][1] * sh2[1][3])
    sh3[5][5] = sh1[1][1] * sh2[4][4] + (sh1[2][1] * sh2[3][4] - sh1[0][1] * sh2[1][4])
    sh3[5][6] = kSqrt01_06 * (sh1[1][2] * sh2[4][4] - sh1[1][0] * sh2[4][0]) + kSqrt01_06 * (
                (sh1[2][2] * sh2[3][4] - sh1[2][0] * sh2[3][0]) - (sh1[0][2] * sh2[1][4] - sh1[0][0] * sh2[1][0]))

    sh3[6][0] = kSqrt01_04 * (
                (sh1[2][2] * sh2[4][0] + sh1[2][0] * sh2[4][4]) - (sh1[0][2] * sh2[0][0] + sh1[0][0] * sh2[0][4]))
    sh3[6][1] = kSqrt03_02 * (sh1[2][1] * sh2[4][0] - sh1[0][1] * sh2[0][0])
    sh3[6][2] = kSqrt15_16 * (sh1[2][1] * sh2[4][1] - sh1[0][1] * sh2[0][1])
    sh3[6][3] = kSqrt05_06 * (sh1[2][1] * sh2[4][2] - sh1[0][1] * sh2[0][2])
    sh3[6][4] = kSqrt15_16 * (sh1[2][1] * sh2[4][3] - sh1[0][1] * sh2[0][3])
    sh3[6][5] = kSqrt03_02 * (sh1[2][1] * sh2[4][4] - sh1[0][1] * sh2[0][4])
    sh3[6][6] = kSqrt01_04 * (
                (sh1[2][2] * sh2[4][4] - sh1[2][0] * sh2[4][0]) - (sh1[0][2] * sh2[0][4] - sh1[0][0] * sh2[0][0]))

    return sh3


def get_sh4(sh1, sh2, sh3):
    sh4 = np.zeros((9, 9), dtype=sh1.dtype)

    sh4[0][0] = kSqrt01_04 * (
                (sh1[2][2] * sh3[0][0] + sh1[2][0] * sh3[0][6]) + (sh1[0][2] * sh3[6][0] + sh1[0][0] * sh3[6][6]))
    sh4[0][1] = kSqrt02_01 * (sh1[2][1] * sh3[0][0] + sh1[0][1] * sh3[6][0])
    sh4[0][2] = kSqrt07_06 * (sh1[2][1] * sh3[0][1] + sh1[0][1] * sh3[6][1])
    sh4[0][3] = kSqrt14_15 * (sh1[2][1] * sh3[0][2] + sh1[0][1] * sh3[6][2])
    sh4[0][4] = kSqrt07_08 * (sh1[2][1] * sh3[0][3] + sh1[0][1] * sh3[6][3])
    sh4[0][5] = kSqrt14_15 * (sh1[2][1] * sh3[0][4] + sh1[0][1] * sh3[6][4])
    sh4[0][6] = kSqrt07_06 * (sh1[2][1] * sh3[0][5] + sh1[0][1] * sh3[6][5])
    sh4[0][7] = kSqrt02_01 * (sh1[2][1] * sh3[0][6] + sh1[0][1] * sh3[6][6])
    sh4[0][8] = kSqrt01_04 * (
                (sh1[2][2] * sh3[0][6] - sh1[2][0] * sh3[0][0]) + (sh1[0][2] * sh3[6][6] - sh1[0][0] * sh3[6][0]))

    sh4[1][0] = kSqrt01_08 * (sh1[1][2] * sh3[0][0] + sh1[1][0] * sh3[0][6]) + kSqrt03_16 * (
                (sh1[2][2] * sh3[1][0] + sh1[2][0] * sh3[1][6]) + (sh1[0][2] * sh3[5][0] + sh1[0][0] * sh3[5][6]))
    sh4[1][1] = sh1[1][1] * sh3[0][0] + kSqrt03_02 * (sh1[2][1] * sh3[1][0] + sh1[0][1] * sh3[5][0])
    sh4[1][2] = kSqrt07_12 * sh1[1][1] * sh3[0][1] + kSqrt07_08 * (sh1[2][1] * sh3[1][1] + sh1[0][1] * sh3[5][1])
    sh4[1][3] = kSqrt07_15 * sh1[1][1] * sh3[0][2] + kSqrt07_10 * (sh1[2][1] * sh3[1][2] + sh1[0][1] * sh3[5][2])
    sh4[1][4] = kSqrt07_16 * sh1[1][1] * sh3[0][3] + kSqrt21_32 * (sh1[2][1] * sh3[1][3] + sh1[0][1] * sh3[5][3])
    sh4[1][5] = kSqrt07_15 * sh1[1][1] * sh3[0][4] + kSqrt07_10 * (sh1[2][1] * sh3[1][4] + sh1[0][1] * sh3[5][4])
    sh4[1][6] = kSqrt07_12 * sh1[1][1] * sh3[0][5] + kSqrt07_08 * (sh1[2][1] * sh3[1][5] + sh1[0][1] * sh3[5][5])
    sh4[1][7] = sh1[1][1] * sh3[0][6] + kSqrt03_02 * (sh1[2][1] * sh3[1][6] + sh1[0][1] * sh3[5][6])
    sh4[1][8] = kSqrt01_08 * (sh1[1][2] * sh3[0][6] - sh1[1][0] * sh3[0][0]) + kSqrt03_16 * (
                (sh1[2][2] * sh3[1][6] - sh1[2][0] * sh3[1][0]) + (sh1[0][2] * sh3[5][6] - sh1[0][0] * sh3[5][0]))

    sh4[2][0] = kSqrt03_14 * (sh1[1][2] * sh3[1][0] + sh1[1][0] * sh3[1][6]) + kSqrt15_112 * (
                (sh1[2][2] * sh3[2][0] + sh1[2][0] * sh3[2][6]) + (
                    sh1[0][2] * sh3[4][0] + sh1[0][0] * sh3[4][6])) - kSqrt01_112 * (
                            (sh1[2][2] * sh3[0][0] + sh1[2][0] * sh3[0][6]) - (
                                sh1[0][2] * sh3[6][0] + sh1[0][0] * sh3[6][6]))
    sh4[2][1] = kSqrt12_07 * sh1[1][1] * sh3[1][0] + kSqrt15_14 * (
                sh1[2][1] * sh3[2][0] + sh1[0][1] * sh3[4][0]) - kSqrt01_14 * (
                            sh1[2][1] * sh3[0][0] - sh1[0][1] * sh3[6][0])
    sh4[2][2] = sh1[1][1] * sh3[1][1] + kSqrt05_08 * (sh1[2][1] * sh3[2][1] + sh1[0][1] * sh3[4][1]) - kSqrt01_24 * (
                sh1[2][1] * sh3[0][1] - sh1[0][1] * sh3[6][1])
    sh4[2][3] = kSqrt04_05 * sh1[1][1] * sh3[1][2] + kSqrt01_02 * (
                sh1[2][1] * sh3[2][2] + sh1[0][1] * sh3[4][2]) - kSqrt01_30 * (
                            sh1[2][1] * sh3[0][2] - sh1[0][1] * sh3[6][2])
    sh4[2][4] = kSqrt03_04 * sh1[1][1] * sh3[1][3] + kSqrt15_32 * (
                sh1[2][1] * sh3[2][3] + sh1[0][1] * sh3[4][3]) - kSqrt01_32 * (
                            sh1[2][1] * sh3[0][3] - sh1[0][1] * sh3[6][3])
    sh4[2][5] = kSqrt04_05 * sh1[1][1] * sh3[1][4] + kSqrt01_02 * (
                sh1[2][1] * sh3[2][4] + sh1[0][1] * sh3[4][4]) - kSqrt01_30 * (
                            sh1[2][1] * sh3[0][4] - sh1[0][1] * sh3[6][4])
    sh4[2][6] = sh1[1][1] * sh3[1][5] + kSqrt05_08 * (sh1[2][1] * sh3[2][5] + sh1[0][1] * sh3[4][5]) - kSqrt01_24 * (
                sh1[2][1] * sh3[0][5] - sh1[0][1] * sh3[6][5])
    sh4[2][7] = kSqrt12_07 * sh1[1][1] * sh3[1][6] + kSqrt15_14 * (
                sh1[2][1] * sh3[2][6] + sh1[0][1] * sh3[4][6]) - kSqrt01_14 * (
                            sh1[2][1] * sh3[0][6] - sh1[0][1] * sh3[6][6])
    sh4[2][8] = kSqrt03_14 * (sh1[1][2] * sh3[1][6] - sh1[1][0] * sh3[1][0]) + kSqrt15_112 * (
                (sh1[2][2] * sh3[2][6] - sh1[2][0] * sh3[2][0]) + (
                    sh1[0][2] * sh3[4][6] - sh1[0][0] * sh3[4][0])) - kSqrt01_112 * (
                            (sh1[2][2] * sh3[0][6] - sh1[2][0] * sh3[0][0]) - (
                                sh1[0][2] * sh3[6][6] - sh1[0][0] * sh3[6][0]))

    sh4[3][0] = kSqrt15_56 * (sh1[1][2] * sh3[2][0] + sh1[1][0] * sh3[2][6]) + kSqrt05_28 * (
                sh1[0][2] * sh3[3][0] + sh1[0][0] * sh3[3][6]) - kSqrt03_112 * (
                            (sh1[2][2] * sh3[1][0] + sh1[2][0] * sh3[1][6]) - (
                                sh1[0][2] * sh3[5][0] + sh1[0][0] * sh3[5][6]))
    sh4[3][1] = kSqrt15_07 * sh1[1][1] * sh3[2][0] + kSqrt10_07 * sh1[0][1] * sh3[3][0] - kSqrt03_14 * (
                sh1[2][1] * sh3[1][0] - sh1[0][1] * sh3[5][0])
    sh4[3][2] = kSqrt05_04 * sh1[1][1] * sh3[2][1] + kSqrt05_06 * sh1[0][1] * sh3[3][1] - kSqrt01_08 * (
                sh1[2][1] * sh3[1][1] - sh1[0][1] * sh3[5][1])
    sh4[3][3] = sh1[1][1] * sh3[2][2] + kSqrt02_03 * sh1[0][1] * sh3[3][2] - kSqrt01_10 * (
                sh1[2][1] * sh3[1][2] - sh1[0][1] * sh3[5][2])
    sh4[3][4] = kSqrt15_16 * sh1[1][1] * sh3[2][3] + kSqrt05_08 * sh1[0][1] * sh3[3][3] - kSqrt03_32 * (
                sh1[2][1] * sh3[1][3] - sh1[0][1] * sh3[5][3])
    sh4[3][5] = sh1[1][1] * sh3[2][4] + kSqrt02_03 * sh1[0][1] * sh3[3][4] - kSqrt01_10 * (
                sh1[2][1] * sh3[1][4] - sh1[0][1] * sh3[5][4])
    sh4[3][6] = kSqrt05_04 * sh1[1][1] * sh3[2][5] + kSqrt05_06 * sh1[0][1] * sh3[3][5] - kSqrt01_08 * (
                sh1[2][1] * sh3[1][5] - sh1[0][1] * sh3[5][5])
    sh4[3][7] = kSqrt15_07 * sh1[1][1] * sh3[2][6] + kSqrt10_07 * sh1[0][1] * sh3[3][6] - kSqrt03_14 * (
                sh1[2][1] * sh3[1][6] - sh1[0][1] * sh3[5][6])
    sh4[3][8] = kSqrt15_56 * (sh1[1][2] * sh3[2][6] - sh1[1][0] * sh3[2][0]) + kSqrt05_28 * (
                sh1[0][2] * sh3[3][6] - sh1[0][0] * sh3[3][0]) - kSqrt03_112 * (
                            (sh1[2][2] * sh3[1][6] - sh1[2][0] * sh3[1][0]) - (
                                sh1[0][2] * sh3[5][6] - sh1[0][0] * sh3[5][0]))

    sh4[4][0] = kSqrt02_07 * (sh1[1][2] * sh3[3][0] + sh1[1][0] * sh3[3][6]) - kSqrt03_28 * (
                (sh1[2][2] * sh3[4][0] + sh1[2][0] * sh3[4][6]) + (sh1[0][2] * sh3[2][0] + sh1[0][0] * sh3[2][6]))
    sh4[4][1] = kSqrt16_07 * sh1[1][1] * sh3[3][0] - kSqrt06_07 * (sh1[2][1] * sh3[4][0] + sh1[0][1] * sh3[2][0])
    sh4[4][2] = kSqrt04_03 * sh1[1][1] * sh3[3][1] - kSqrt01_02 * (sh1[2][1] * sh3[4][1] + sh1[0][1] * sh3[2][1])
    sh4[4][3] = kSqrt16_15 * sh1[1][1] * sh3[3][2] - kSqrt02_05 * (sh1[2][1] * sh3[4][2] + sh1[0][1] * sh3[2][2])
    sh4[4][4] = sh1[1][1] * sh3[3][3] - kSqrt03_08 * (sh1[2][1] * sh3[4][3] + sh1[0][1] * sh3[2][3])
    sh4[4][5] = kSqrt16_15 * sh1[1][1] * sh3[3][4] - kSqrt02_05 * (sh1[2][1] * sh3[4][4] + sh1[0][1] * sh3[2][4])
    sh4[4][6] = kSqrt04_03 * sh1[1][1] * sh3[3][5] - kSqrt01_02 * (sh1[2][1] * sh3[4][5] + sh1[0][1] * sh3[2][5])
    sh4[4][7] = kSqrt16_07 * sh1[1][1] * sh3[3][6] - kSqrt06_07 * (sh1[2][1] * sh3[4][6] + sh1[0][1] * sh3[2][6])
    sh4[4][8] = kSqrt02_07 * (sh1[1][2] * sh3[3][6] - sh1[1][0] * sh3[3][0]) - kSqrt03_28 * (
                (sh1[2][2] * sh3[4][6] - sh1[2][0] * sh3[4][0]) + (sh1[0][2] * sh3[2][6] - sh1[0][0] * sh3[2][0]))

    sh4[5][0] = kSqrt15_56 * (sh1[1][2] * sh3[4][0] + sh1[1][0] * sh3[4][6]) + kSqrt05_28 * (
                sh1[2][2] * sh3[3][0] + sh1[2][0] * sh3[3][6]) - kSqrt03_112 * (
                            (sh1[2][2] * sh3[5][0] + sh1[2][0] * sh3[5][6]) + (
                                sh1[0][2] * sh3[1][0] + sh1[0][0] * sh3[1][6]))
    sh4[5][1] = kSqrt15_07 * sh1[1][1] * sh3[4][0] + kSqrt10_07 * sh1[2][1] * sh3[3][0] - kSqrt03_14 * (
                sh1[2][1] * sh3[5][0] + sh1[0][1] * sh3[1][0])
    sh4[5][2] = kSqrt05_04 * sh1[1][1] * sh3[4][1] + kSqrt05_06 * sh1[2][1] * sh3[3][1] - kSqrt01_08 * (
                sh1[2][1] * sh3[5][1] + sh1[0][1] * sh3[1][1])
    sh4[5][3] = sh1[1][1] * sh3[4][2] + kSqrt02_03 * sh1[2][1] * sh3[3][2] - kSqrt01_10 * (
                sh1[2][1] * sh3[5][2] + sh1[0][1] * sh3[1][2])
    sh4[5][4] = kSqrt15_16 * sh1[1][1] * sh3[4][3] + kSqrt05_08 * sh1[2][1] * sh3[3][3] - kSqrt03_32 * (
                sh1[2][1] * sh3[5][3] + sh1[0][1] * sh3[1][3])
    sh4[5][5] = sh1[1][1] * sh3[4][4] + kSqrt02_03 * sh1[2][1] * sh3[3][4] - kSqrt01_10 * (
                sh1[2][1] * sh3[5][4] + sh1[0][1] * sh3[1][4])
    sh4[5][6] = kSqrt05_04 * sh1[1][1] * sh3[4][5] + kSqrt05_06 * sh1[2][1] * sh3[3][5] - kSqrt01_08 * (
                sh1[2][1] * sh3[5][5] + sh1[0][1] * sh3[1][5])
    sh4[5][7] = kSqrt15_07 * sh1[1][1] * sh3[4][6] + kSqrt10_07 * sh1[2][1] * sh3[3][6] - kSqrt03_14 * (
                sh1[2][1] * sh3[5][6] + sh1[0][1] * sh3[1][6])
    sh4[5][8] = kSqrt15_56 * (sh1[1][2] * sh3[4][6] - sh1[1][0] * sh3[4][0]) + kSqrt05_28 * (
                sh1[2][2] * sh3[3][6] - sh1[2][0] * sh3[3][0]) - kSqrt03_112 * (
                            (sh1[2][2] * sh3[5][6] - sh1[2][0] * sh3[5][0]) + (
                                sh1[0][2] * sh3[1][6] - sh1[0][0] * sh3[1][0]))

    sh4[6][0] = kSqrt03_14 * (sh1[1][2] * sh3[5][0] + sh1[1][0] * sh3[5][6]) + kSqrt15_112 * (
                (sh1[2][2] * sh3[4][0] + sh1[2][0] * sh3[4][6]) - (
                    sh1[0][2] * sh3[2][0] + sh1[0][0] * sh3[2][6])) - kSqrt01_112 * (
                            (sh1[2][2] * sh3[6][0] + sh1[2][0] * sh3[6][6]) + (
                                sh1[0][2] * sh3[0][0] + sh1[0][0] * sh3[0][6]))
    sh4[6][1] = kSqrt12_07 * sh1[1][1] * sh3[5][0] + kSqrt15_14 * (
                sh1[2][1] * sh3[4][0] - sh1[0][1] * sh3[2][0]) - kSqrt01_14 * (
                            sh1[2][1] * sh3[6][0] + sh1[0][1] * sh3[0][0])
    sh4[6][2] = sh1[1][1] * sh3[5][1] + kSqrt05_08 * (sh1[2][1] * sh3[4][1] - sh1[0][1] * sh3[2][1]) - kSqrt01_24 * (
                sh1[2][1] * sh3[6][1] + sh1[0][1] * sh3[0][1])
    sh4[6][3] = kSqrt04_05 * sh1[1][1] * sh3[5][2] + kSqrt01_02 * (
                sh1[2][1] * sh3[4][2] - sh1[0][1] * sh3[2][2]) - kSqrt01_30 * (
                            sh1[2][1] * sh3[6][2] + sh1[0][1] * sh3[0][2])
    sh4[6][4] = kSqrt03_04 * sh1[1][1] * sh3[5][3] + kSqrt15_32 * (
                sh1[2][1] * sh3[4][3] - sh1[0][1] * sh3[2][3]) - kSqrt01_32 * (
                            sh1[2][1] * sh3[6][3] + sh1[0][1] * sh3[0][3])
    sh4[6][5] = kSqrt04_05 * sh1[1][1] * sh3[5][4] + kSqrt01_02 * (
                sh1[2][1] * sh3[4][4] - sh1[0][1] * sh3[2][4]) - kSqrt01_30 * (
                            sh1[2][1] * sh3[6][4] + sh1[0][1] * sh3[0][4])
    sh4[6][6] = sh1[1][1] * sh3[5][5] + kSqrt05_08 * (sh1[2][1] * sh3[4][5] - sh1[0][1] * sh3[2][5]) - kSqrt01_24 * (
                sh1[2][1] * sh3[6][5] + sh1[0][1] * sh3[0][5])
    sh4[6][7] = kSqrt12_07 * sh1[1][1] * sh3[5][6] + kSqrt15_14 * (
                sh1[2][1] * sh3[4][6] - sh1[0][1] * sh3[2][6]) - kSqrt01_14 * (
                            sh1[2][1] * sh3[6][6] + sh1[0][1] * sh3[0][6])
    sh4[6][8] = kSqrt03_14 * (sh1[1][2] * sh3[5][6] - sh1[1][0] * sh3[5][0]) + kSqrt15_112 * (
                (sh1[2][2] * sh3[4][6] - sh1[2][0] * sh3[4][0]) - (
                    sh1[0][2] * sh3[2][6] - sh1[0][0] * sh3[2][0])) - kSqrt01_112 * (
                            (sh1[2][2] * sh3[6][6] - sh1[2][0] * sh3[6][0]) + (
                                sh1[0][2] * sh3[0][6] - sh1[0][0] * sh3[0][0]))

    sh4[7][0] = kSqrt01_08 * (sh1[1][2] * sh3[6][0] + sh1[1][0] * sh3[6][6]) + kSqrt03_16 * (
                (sh1[2][2] * sh3[5][0] + sh1[2][0] * sh3[5][6]) - (sh1[0][2] * sh3[1][0] + sh1[0][0] * sh3[1][6]))
    sh4[7][1] = sh1[1][1] * sh3[6][0] + kSqrt03_02 * (sh1[2][1] * sh3[5][0] - sh1[0][1] * sh3[1][0])
    sh4[7][2] = kSqrt07_12 * sh1[1][1] * sh3[6][1] + kSqrt07_08 * (sh1[2][1] * sh3[5][1] - sh1[0][1] * sh3[1][1])
    sh4[7][3] = kSqrt07_15 * sh1[1][1] * sh3[6][2] + kSqrt07_10 * (sh1[2][1] * sh3[5][2] - sh1[0][1] * sh3[1][2])
    sh4[7][4] = kSqrt07_16 * sh1[1][1] * sh3[6][3] + kSqrt21_32 * (sh1[2][1] * sh3[5][3] - sh1[0][1] * sh3[1][3])
    sh4[7][5] = kSqrt07_15 * sh1[1][1] * sh3[6][4] + kSqrt07_10 * (sh1[2][1] * sh3[5][4] - sh1[0][1] * sh3[1][4])
    sh4[7][6] = kSqrt07_12 * sh1[1][1] * sh3[6][5] + kSqrt07_08 * (sh1[2][1] * sh3[5][5] - sh1[0][1] * sh3[1][5])
    sh4[7][7] = sh1[1][1] * sh3[6][6] + kSqrt03_02 * (sh1[2][1] * sh3[5][6] - sh1[0][1] * sh3[1][6])
    sh4[7][8] = kSqrt01_08 * (sh1[1][2] * sh3[6][6] - sh1[1][0] * sh3[6][0]) + kSqrt03_16 * (
                (sh1[2][2] * sh3[5][6] - sh1[2][0] * sh3[5][0]) - (sh1[0][2] * sh3[1][6] - sh1[0][0] * sh3[1][0]))

    sh4[8][0] = kSqrt01_04 * (
                (sh1[2][2] * sh3[6][0] + sh1[2][0] * sh3[6][6]) - (sh1[0][2] * sh3[0][0] + sh1[0][0] * sh3[0][6]))
    sh4[8][1] = kSqrt02_01 * (sh1[2][1] * sh3[6][0] - sh1[0][1] * sh3[0][0])
    sh4[8][2] = kSqrt07_06 * (sh1[2][1] * sh3[6][1] - sh1[0][1] * sh3[0][1])
    sh4[8][3] = kSqrt14_15 * (sh1[2][1] * sh3[6][2] - sh1[0][1] * sh3[0][2])
    sh4[8][4] = kSqrt07_08 * (sh1[2][1] * sh3[6][3] - sh1[0][1] * sh3[0][3])
    sh4[8][5] = kSqrt14_15 * (sh1[2][1] * sh3[6][4] - sh1[0][1] * sh3[0][4])
    sh4[8][6] = kSqrt07_06 * (sh1[2][1] * sh3[6][5] - sh1[0][1] * sh3[0][5])
    sh4[8][7] = kSqrt02_01 * (sh1[2][1] * sh3[6][6] - sh1[0][1] * sh3[0][6])
    sh4[8][8] = kSqrt01_04 * (
                (sh1[2][2] * sh3[6][6] - sh1[2][0] * sh3[6][0]) - (sh1[0][2] * sh3[0][6] - sh1[0][0] * sh3[0][0]))

    return sh4


class SHRotator:
    def __init__(self, R, deg=3):
        self.deg = deg
        if deg > 0: self.sh1 = get_sh1(R)
        if deg > 1: self.sh2 = get_sh2(self.sh1)
        if deg > 2: self.sh3 = get_sh3(self.sh1, self.sh2)
        if deg > 3: self.sh4 = get_sh4(self.sh1, self.sh2, self.sh3)
        if deg > 4: raise NotImplementedError

    def __call__(self, shs_in):
        # shs_in: (n, deg)
        shs_out = []
        # deg 0
        shs_out.append(shs_in[..., 0:1])
        if self.deg > 0:
            shs_out.append((self.sh1 @ shs_in[..., 1:4].T).T)
        if self.deg > 1:
            shs_out.append((self.sh2 @ shs_in[..., 4:9].T).T)
        if self.deg > 2:
            shs_out.append((self.sh3 @ shs_in[..., 9:16].T).T)
        if self.deg > 3:
            shs_out.append((self.sh4 @ shs_in[..., 16:25].T).T)
        if self.deg > 4: raise NotImplementedError
        shs_out = np.concatenate(shs_out, axis=-1)
        assert shs_out.shape == shs_in.shape
        return shs_out


import torch
import numpy as np # Vẫn cần numpy để tạo dữ liệu test ban đầu nếu muốn

# Định nghĩa các hằng số dưới dạng tensor PyTorch
# Nên đặt device và dtype phù hợp khi sử dụng thực tế
# Ví dụ: device='cuda', dtype=torch.float32
_device = None # Hoặc torch.device('cuda') nếu có GPU
_dtype = torch.float32

kSqrt02_01 = torch.sqrt(torch.tensor(2.0 / 1.0, dtype=_dtype, device=_device))
kSqrt01_02 = torch.sqrt(torch.tensor(1.0 / 2.0, dtype=_dtype, device=_device))
kSqrt03_02 = torch.sqrt(torch.tensor(3.0 / 2.0, dtype=_dtype, device=_device))
kSqrt01_03 = torch.sqrt(torch.tensor(1.0 / 3.0, dtype=_dtype, device=_device))
kSqrt02_03 = torch.sqrt(torch.tensor(2.0 / 3.0, dtype=_dtype, device=_device))
kSqrt04_03 = torch.sqrt(torch.tensor(4.0 / 3.0, dtype=_dtype, device=_device))
kSqrt01_04 = torch.sqrt(torch.tensor(1.0 / 4.0, dtype=_dtype, device=_device))
kSqrt03_04 = torch.sqrt(torch.tensor(3.0 / 4.0, dtype=_dtype, device=_device))
kSqrt05_04 = torch.sqrt(torch.tensor(5.0 / 4.0, dtype=_dtype, device=_device))
kSqrt01_05 = torch.sqrt(torch.tensor(1.0 / 5.0, dtype=_dtype, device=_device))
kSqrt02_05 = torch.sqrt(torch.tensor(2.0 / 5.0, dtype=_dtype, device=_device))
kSqrt03_05 = torch.sqrt(torch.tensor(3.0 / 5.0, dtype=_dtype, device=_device))
kSqrt04_05 = torch.sqrt(torch.tensor(4.0 / 5.0, dtype=_dtype, device=_device))
kSqrt06_05 = torch.sqrt(torch.tensor(6.0 / 5.0, dtype=_dtype, device=_device))
kSqrt08_05 = torch.sqrt(torch.tensor(8.0 / 5.0, dtype=_dtype, device=_device))
kSqrt09_05 = torch.sqrt(torch.tensor(9.0 / 5.0, dtype=_dtype, device=_device))
kSqrt01_06 = torch.sqrt(torch.tensor(1.0 / 6.0, dtype=_dtype, device=_device))
kSqrt05_06 = torch.sqrt(torch.tensor(5.0 / 6.0, dtype=_dtype, device=_device))
kSqrt07_06 = torch.sqrt(torch.tensor(7.0 / 6.0, dtype=_dtype, device=_device))
kSqrt02_07 = torch.sqrt(torch.tensor(2.0 / 7.0, dtype=_dtype, device=_device)) # Sửa lỗi số 02.0 thành 2.0
kSqrt06_07 = torch.sqrt(torch.tensor(6.0 / 7.0, dtype=_dtype, device=_device))
kSqrt10_07 = torch.sqrt(torch.tensor(10.0 / 7.0, dtype=_dtype, device=_device))
kSqrt12_07 = torch.sqrt(torch.tensor(12.0 / 7.0, dtype=_dtype, device=_device))
kSqrt15_07 = torch.sqrt(torch.tensor(15.0 / 7.0, dtype=_dtype, device=_device))
kSqrt16_07 = torch.sqrt(torch.tensor(16.0 / 7.0, dtype=_dtype, device=_device))
kSqrt01_08 = torch.sqrt(torch.tensor(1.0 / 8.0, dtype=_dtype, device=_device))
kSqrt03_08 = torch.sqrt(torch.tensor(3.0 / 8.0, dtype=_dtype, device=_device))
kSqrt05_08 = torch.sqrt(torch.tensor(5.0 / 8.0, dtype=_dtype, device=_device))
kSqrt07_08 = torch.sqrt(torch.tensor(7.0 / 8.0, dtype=_dtype, device=_device))
kSqrt09_08 = torch.sqrt(torch.tensor(9.0 / 8.0, dtype=_dtype, device=_device))
kSqrt05_09 = torch.sqrt(torch.tensor(5.0 / 9.0, dtype=_dtype, device=_device))
kSqrt08_09 = torch.sqrt(torch.tensor(8.0 / 9.0, dtype=_dtype, device=_device))
kSqrt01_10 = torch.sqrt(torch.tensor(1.0 / 10.0, dtype=_dtype, device=_device))
kSqrt03_10 = torch.sqrt(torch.tensor(3.0 / 10.0, dtype=_dtype, device=_device))
kSqrt07_10 = torch.sqrt(torch.tensor(7.0 / 10.0, dtype=_dtype, device=_device))
kSqrt09_10 = torch.sqrt(torch.tensor(9.0 / 10.0, dtype=_dtype, device=_device))
kSqrt01_12 = torch.sqrt(torch.tensor(1.0 / 12.0, dtype=_dtype, device=_device))
kSqrt07_12 = torch.sqrt(torch.tensor(7.0 / 12.0, dtype=_dtype, device=_device))
kSqrt11_12 = torch.sqrt(torch.tensor(11.0 / 12.0, dtype=_dtype, device=_device))
kSqrt01_14 = torch.sqrt(torch.tensor(1.0 / 14.0, dtype=_dtype, device=_device))
kSqrt03_14 = torch.sqrt(torch.tensor(3.0 / 14.0, dtype=_dtype, device=_device))
kSqrt15_14 = torch.sqrt(torch.tensor(15.0 / 14.0, dtype=_dtype, device=_device))
kSqrt04_15 = torch.sqrt(torch.tensor(4.0 / 15.0, dtype=_dtype, device=_device))
kSqrt07_15 = torch.sqrt(torch.tensor(7.0 / 10.0, dtype=_dtype, device=_device)) # Chú ý: Hằng số này có vẻ khác với tên (10 thay vì 15)
kSqrt14_15 = torch.sqrt(torch.tensor(14.0 / 15.0, dtype=_dtype, device=_device))
kSqrt16_15 = torch.sqrt(torch.tensor(16.0 / 15.0, dtype=_dtype, device=_device))
kSqrt01_16 = torch.sqrt(torch.tensor(1.0 / 16.0, dtype=_dtype, device=_device))
kSqrt03_16 = torch.sqrt(torch.tensor(3.0 / 16.0, dtype=_dtype, device=_device))
kSqrt07_16 = torch.sqrt(torch.tensor(7.0 / 16.0, dtype=_dtype, device=_device))
kSqrt15_16 = torch.sqrt(torch.tensor(15.0 / 16.0, dtype=_dtype, device=_device))
kSqrt01_18 = torch.sqrt(torch.tensor(1.0 / 18.0, dtype=_dtype, device=_device))
kSqrt01_24 = torch.sqrt(torch.tensor(1.0 / 24.0, dtype=_dtype, device=_device))
kSqrt03_25 = torch.sqrt(torch.tensor(3.0 / 25.0, dtype=_dtype, device=_device))
kSqrt09_25 = torch.sqrt(torch.tensor(9.0 / 25.0, dtype=_dtype, device=_device))
kSqrt14_25 = torch.sqrt(torch.tensor(14.0 / 25.0, dtype=_dtype, device=_device))
kSqrt16_25 = torch.sqrt(torch.tensor(16.0 / 25.0, dtype=_dtype, device=_device))
kSqrt18_25 = torch.sqrt(torch.tensor(18.0 / 25.0, dtype=_dtype, device=_device))
kSqrt21_25 = torch.sqrt(torch.tensor(21.0 / 25.0, dtype=_dtype, device=_device))
kSqrt24_25 = torch.sqrt(torch.tensor(24.0 / 25.0, dtype=_dtype, device=_device))
kSqrt03_28 = torch.sqrt(torch.tensor(3.0 / 28.0, dtype=_dtype, device=_device))
kSqrt05_28 = torch.sqrt(torch.tensor(5.0 / 28.0, dtype=_dtype, device=_device))
kSqrt01_30 = torch.sqrt(torch.tensor(1.0 / 30.0, dtype=_dtype, device=_device))
kSqrt01_32 = torch.sqrt(torch.tensor(1.0 / 32.0, dtype=_dtype, device=_device))
kSqrt03_32 = torch.sqrt(torch.tensor(3.0 / 32.0, dtype=_dtype, device=_device))
kSqrt15_32 = torch.sqrt(torch.tensor(15.0 / 32.0, dtype=_dtype, device=_device))
kSqrt21_32 = torch.sqrt(torch.tensor(21.0 / 32.0, dtype=_dtype, device=_device))
kSqrt11_36 = torch.sqrt(torch.tensor(11.0 / 36.0, dtype=_dtype, device=_device))
kSqrt35_36 = torch.sqrt(torch.tensor(35.0 / 36.0, dtype=_dtype, device=_device))
kSqrt01_50 = torch.sqrt(torch.tensor(1.0 / 50.0, dtype=_dtype, device=_device))
kSqrt03_50 = torch.sqrt(torch.tensor(3.0 / 50.0, dtype=_dtype, device=_device))
kSqrt21_50 = torch.sqrt(torch.tensor(21.0 / 50.0, dtype=_dtype, device=_device))
kSqrt15_56 = torch.sqrt(torch.tensor(15.0 / 56.0, dtype=_dtype, device=_device))
kSqrt01_60 = torch.sqrt(torch.tensor(1.0 / 60.0, dtype=_dtype, device=_device))
kSqrt01_112 = torch.sqrt(torch.tensor(1.0 / 112.0, dtype=_dtype, device=_device))
kSqrt03_112 = torch.sqrt(torch.tensor(3.0 / 112.0, dtype=_dtype, device=_device))
kSqrt15_112 = torch.sqrt(torch.tensor(15.0 / 112.0, dtype=_dtype, device=_device))


def get_sh1_torch(R):
    """
    Tính toán ma trận xoay SH bậc 1 từ ma trận xoay 3x3.
    Args:
        R (torch.Tensor): Ma trận xoay (3, 3).
    Returns:
        torch.Tensor: Ma trận xoay SH bậc 1 (3, 3).
    """
    # torch.roll hoạt động tương tự np.roll
    # dims=(-1,) nghĩa là roll cột cuối cùng
    # dims=(-2,) nghĩa là roll hàng cuối cùng
    return torch.roll(torch.roll(R, shifts=(-1,), dims=(-1)), shifts=(-1,), dims=(-2))

def get_sh2_torch(sh1):
    """
    Tính toán ma trận xoay SH bậc 2.
    Args:
        sh1 (torch.Tensor): Ma trận xoay SH bậc 1 (3, 3).
    Returns:
        torch.Tensor: Ma trận xoay SH bậc 2 (5, 5).
    """
    sh2 = torch.zeros((5, 5), dtype=sh1.dtype, device=sh1.device)

    # Chuyển đổi trực tiếp các phép tính, sử dụng indexing của PyTorch
    # Lưu ý: indexing [i][j] của numpy tương đương với [i, j] trong PyTorch
    sh2[0, 0] = kSqrt01_04 * ((sh1[2, 2] * sh1[0, 0] + sh1[2, 0] * sh1[0, 2]) + (sh1[0, 2] * sh1[2, 0] + sh1[0, 0] * sh1[2, 2]))
    sh2[0, 1] = (sh1[2, 1] * sh1[0, 0] + sh1[0, 1] * sh1[2, 0])
    sh2[0, 2] = kSqrt03_04 * (sh1[2, 1] * sh1[0, 1] + sh1[0, 1] * sh1[2, 1])
    sh2[0, 3] = (sh1[2, 1] * sh1[0, 2] + sh1[0, 1] * sh1[2, 2])
    sh2[0, 4] = kSqrt01_04 * ((sh1[2, 2] * sh1[0, 2] - sh1[2, 0] * sh1[0, 0]) + (sh1[0, 2] * sh1[2, 2] - sh1[0, 0] * sh1[2, 0]))

    sh2[1, 0] = kSqrt01_04 * ((sh1[1, 2] * sh1[0, 0] + sh1[1, 0] * sh1[0, 2]) + (sh1[0, 2] * sh1[1, 0] + sh1[0, 0] * sh1[1, 2]))
    sh2[1, 1] = sh1[1, 1] * sh1[0, 0] + sh1[0, 1] * sh1[1, 0]
    sh2[1, 2] = kSqrt03_04 * (sh1[1, 1] * sh1[0, 1] + sh1[0, 1] * sh1[1, 1])
    sh2[1, 3] = sh1[1, 1] * sh1[0, 2] + sh1[0, 1] * sh1[1, 2]
    sh2[1, 4] = kSqrt01_04 * ((sh1[1, 2] * sh1[0, 2] - sh1[1, 0] * sh1[0, 0]) + (sh1[0, 2] * sh1[1, 2] - sh1[0, 0] * sh1[1, 0]))

    sh2[2, 0] = kSqrt01_03 * (sh1[1, 2] * sh1[1, 0] + sh1[1, 0] * sh1[1, 2]) - kSqrt01_12 * ((sh1[2, 2] * sh1[2, 0] + sh1[2, 0] * sh1[2, 2]) + (sh1[0, 2] * sh1[0, 0] + sh1[0, 0] * sh1[0, 2]))
    sh2[2, 1] = kSqrt04_03 * sh1[1, 1] * sh1[1, 0] - kSqrt01_03 * (sh1[2, 1] * sh1[2, 0] + sh1[0, 1] * sh1[0, 0])
    sh2[2, 2] = sh1[1, 1] * sh1[1, 1] - kSqrt01_04 * (sh1[2, 1] * sh1[2, 1] + sh1[0, 1] * sh1[0, 1])
    sh2[2, 3] = kSqrt04_03 * sh1[1, 1] * sh1[1, 2] - kSqrt01_03 * (sh1[2, 1] * sh1[2, 2] + sh1[0, 1] * sh1[0, 2])
    sh2[2, 4] = kSqrt01_03 * (sh1[1, 2] * sh1[1, 2] - sh1[1, 0] * sh1[1, 0]) - kSqrt01_12 * ((sh1[2, 2] * sh1[2, 2] - sh1[2, 0] * sh1[2, 0]) + (sh1[0, 2] * sh1[0, 2] - sh1[0, 0] * sh1[0, 0]))

    sh2[3, 0] = kSqrt01_04 * ((sh1[1, 2] * sh1[2, 0] + sh1[1, 0] * sh1[2, 2]) + (sh1[2, 2] * sh1[1, 0] + sh1[2, 0] * sh1[1, 2]))
    sh2[3, 1] = sh1[1, 1] * sh1[2, 0] + sh1[2, 1] * sh1[1, 0]
    sh2[3, 2] = kSqrt03_04 * (sh1[1, 1] * sh1[2, 1] + sh1[2, 1] * sh1[1, 1])
    sh2[3, 3] = sh1[1, 1] * sh1[2, 2] + sh1[2, 1] * sh1[1, 2]
    sh2[3, 4] = kSqrt01_04 * ((sh1[1, 2] * sh1[2, 2] - sh1[1, 0] * sh1[2, 0]) + (sh1[2, 2] * sh1[1, 2] - sh1[2, 0] * sh1[1, 0]))

    sh2[4, 0] = kSqrt01_04 * ((sh1[2, 2] * sh1[2, 0] + sh1[2, 0] * sh1[2, 2]) - (sh1[0, 2] * sh1[0, 0] + sh1[0, 0] * sh1[0, 2]))
    sh2[4, 1] = (sh1[2, 1] * sh1[2, 0] - sh1[0, 1] * sh1[0, 0])
    sh2[4, 2] = kSqrt03_04 * (sh1[2, 1] * sh1[2, 1] - sh1[0, 1] * sh1[0, 1])
    sh2[4, 3] = (sh1[2, 1] * sh1[2, 2] - sh1[0, 1] * sh1[0, 2])
    sh2[4, 4] = kSqrt01_04 * ((sh1[2, 2] * sh1[2, 2] - sh1[2, 0] * sh1[2, 0]) - (sh1[0, 2] * sh1[0, 2] - sh1[0, 0] * sh1[0, 0]))

    return sh2

def get_sh3_torch(sh1, sh2):
    """
    Tính toán ma trận xoay SH bậc 3.
    Args:
        sh1 (torch.Tensor): Ma trận xoay SH bậc 1 (3, 3).
        sh2 (torch.Tensor): Ma trận xoay SH bậc 2 (5, 5).
    Returns:
        torch.Tensor: Ma trận xoay SH bậc 3 (7, 7).
    """
    sh3 = torch.zeros((7, 7), dtype=sh1.dtype, device=sh1.device)

    sh3[0, 0] = kSqrt01_04 * ((sh1[2, 2] * sh2[0, 0] + sh1[2, 0] * sh2[0, 4]) + (sh1[0, 2] * sh2[4, 0] + sh1[0, 0] * sh2[4, 4]))
    sh3[0, 1] = kSqrt03_02 * (sh1[2, 1] * sh2[0, 0] + sh1[0, 1] * sh2[4, 0])
    sh3[0, 2] = kSqrt15_16 * (sh1[2, 1] * sh2[0, 1] + sh1[0, 1] * sh2[4, 1])
    sh3[0, 3] = kSqrt05_06 * (sh1[2, 1] * sh2[0, 2] + sh1[0, 1] * sh2[4, 2])
    sh3[0, 4] = kSqrt15_16 * (sh1[2, 1] * sh2[0, 3] + sh1[0, 1] * sh2[4, 3])
    sh3[0, 5] = kSqrt03_02 * (sh1[2, 1] * sh2[0, 4] + sh1[0, 1] * sh2[4, 4])
    sh3[0, 6] = kSqrt01_04 * ((sh1[2, 2] * sh2[0, 4] - sh1[2, 0] * sh2[0, 0]) + (sh1[0, 2] * sh2[4, 4] - sh1[0, 0] * sh2[4, 0]))

    sh3[1, 0] = kSqrt01_06 * (sh1[1, 2] * sh2[0, 0] + sh1[1, 0] * sh2[0, 4]) + kSqrt01_06 * ((sh1[2, 2] * sh2[1, 0] + sh1[2, 0] * sh2[1, 4]) + (sh1[0, 2] * sh2[3, 0] + sh1[0, 0] * sh2[3, 4]))
    sh3[1, 1] = sh1[1, 1] * sh2[0, 0] + (sh1[2, 1] * sh2[1, 0] + sh1[0, 1] * sh2[3, 0])
    sh3[1, 2] = kSqrt05_08 * sh1[1, 1] * sh2[0, 1] + kSqrt05_08 * (sh1[2, 1] * sh2[1, 1] + sh1[0, 1] * sh2[3, 1])
    sh3[1, 3] = kSqrt05_09 * sh1[1, 1] * sh2[0, 2] + kSqrt05_09 * (sh1[2, 1] * sh2[1, 2] + sh1[0, 1] * sh2[3, 2])
    sh3[1, 4] = kSqrt05_08 * sh1[1, 1] * sh2[0, 3] + kSqrt05_08 * (sh1[2, 1] * sh2[1, 3] + sh1[0, 1] * sh2[3, 3])
    sh3[1, 5] = sh1[1, 1] * sh2[0, 4] + (sh1[2, 1] * sh2[1, 4] + sh1[0, 1] * sh2[3, 4])
    sh3[1, 6] = kSqrt01_06 * (sh1[1, 2] * sh2[0, 4] - sh1[1, 0] * sh2[0, 0]) + kSqrt01_06 * ((sh1[2, 2] * sh2[1, 4] - sh1[2, 0] * sh2[1, 0]) + (sh1[0, 2] * sh2[3, 4] - sh1[0, 0] * sh2[3, 0]))

    sh3[2, 0] = kSqrt04_15 * (sh1[1, 2] * sh2[1, 0] + sh1[1, 0] * sh2[1, 4]) + kSqrt01_05 * (sh1[0, 2] * sh2[2, 0] + sh1[0, 0] * sh2[2, 4]) - kSqrt01_60 * ((sh1[2, 2] * sh2[0, 0] + sh1[2, 0] * sh2[0, 4]) - (sh1[0, 2] * sh2[4, 0] + sh1[0, 0] * sh2[4, 4]))
    sh3[2, 1] = kSqrt08_05 * sh1[1, 1] * sh2[1, 0] + kSqrt06_05 * sh1[0, 1] * sh2[2, 0] - kSqrt01_10 * (sh1[2, 1] * sh2[0, 0] - sh1[0, 1] * sh2[4, 0])
    sh3[2, 2] = sh1[1, 1] * sh2[1, 1] + kSqrt03_04 * sh1[0, 1] * sh2[2, 1] - kSqrt01_16 * (sh1[2, 1] * sh2[0, 1] - sh1[0, 1] * sh2[4, 1])
    sh3[2, 3] = kSqrt08_09 * sh1[1, 1] * sh2[1, 2] + kSqrt02_03 * sh1[0, 1] * sh2[2, 2] - kSqrt01_18 * (sh1[2, 1] * sh2[0, 2] - sh1[0, 1] * sh2[4, 2])
    sh3[2, 4] = sh1[1, 1] * sh2[1, 3] + kSqrt03_04 * sh1[0, 1] * sh2[2, 3] - kSqrt01_16 * (sh1[2, 1] * sh2[0, 3] - sh1[0, 1] * sh2[4, 3])
    sh3[2, 5] = kSqrt08_05 * sh1[1, 1] * sh2[1, 4] + kSqrt06_05 * sh1[0, 1] * sh2[2, 4] - kSqrt01_10 * (sh1[2, 1] * sh2[0, 4] - sh1[0, 1] * sh2[4, 4])
    sh3[2, 6] = kSqrt04_15 * (sh1[1, 2] * sh2[1, 4] - sh1[1, 0] * sh2[1, 0]) + kSqrt01_05 * (sh1[0, 2] * sh2[2, 4] - sh1[0, 0] * sh2[2, 0]) - kSqrt01_60 * ((sh1[2, 2] * sh2[0, 4] - sh1[2, 0] * sh2[0, 0]) - (sh1[0, 2] * sh2[4, 4] - sh1[0, 0] * sh2[4, 0]))

    sh3[3, 0] = kSqrt03_10 * (sh1[1, 2] * sh2[2, 0] + sh1[1, 0] * sh2[2, 4]) - kSqrt01_10 * ((sh1[2, 2] * sh2[3, 0] + sh1[2, 0] * sh2[3, 4]) + (sh1[0, 2] * sh2[1, 0] + sh1[0, 0] * sh2[1, 4]))
    sh3[3, 1] = kSqrt09_05 * sh1[1, 1] * sh2[2, 0] - kSqrt03_05 * (sh1[2, 1] * sh2[3, 0] + sh1[0, 1] * sh2[1, 0])
    sh3[3, 2] = kSqrt09_08 * sh1[1, 1] * sh2[2, 1] - kSqrt03_08 * (sh1[2, 1] * sh2[3, 1] + sh1[0, 1] * sh2[1, 1])
    sh3[3, 3] = sh1[1, 1] * sh2[2, 2] - kSqrt01_03 * (sh1[2, 1] * sh2[3, 2] + sh1[0, 1] * sh2[1, 2])
    sh3[3, 4] = kSqrt09_08 * sh1[1, 1] * sh2[2, 3] - kSqrt03_08 * (sh1[2, 1] * sh2[3, 3] + sh1[0, 1] * sh2[1, 3])
    sh3[3, 5] = kSqrt09_05 * sh1[1, 1] * sh2[2, 4] - kSqrt03_05 * (sh1[2, 1] * sh2[3, 4] + sh1[0, 1] * sh2[1, 4])
    sh3[3, 6] = kSqrt03_10 * (sh1[1, 2] * sh2[2, 4] - sh1[1, 0] * sh2[2, 0]) - kSqrt01_10 * ((sh1[2, 2] * sh2[3, 4] - sh1[2, 0] * sh2[3, 0]) + (sh1[0, 2] * sh2[1, 4] - sh1[0, 0] * sh2[1, 0]))

    sh3[4, 0] = kSqrt04_15 * (sh1[1, 2] * sh2[3, 0] + sh1[1, 0] * sh2[3, 4]) + kSqrt01_05 * (sh1[2, 2] * sh2[2, 0] + sh1[2, 0] * sh2[2, 4]) - kSqrt01_60 * ((sh1[2, 2] * sh2[4, 0] + sh1[2, 0] * sh2[4, 4]) + (sh1[0, 2] * sh2[0, 0] + sh1[0, 0] * sh2[0, 4]))
    sh3[4, 1] = kSqrt08_05 * sh1[1, 1] * sh2[3, 0] + kSqrt06_05 * sh1[2, 1] * sh2[2, 0] - kSqrt01_10 * (sh1[2, 1] * sh2[4, 0] + sh1[0, 1] * sh2[0, 0])
    sh3[4, 2] = sh1[1, 1] * sh2[3, 1] + kSqrt03_04 * sh1[2, 1] * sh2[2, 1] - kSqrt01_16 * (sh1[2, 1] * sh2[4, 1] + sh1[0, 1] * sh2[0, 1])
    sh3[4, 3] = kSqrt08_09 * sh1[1, 1] * sh2[3, 2] + kSqrt02_03 * sh1[2, 1] * sh2[2, 2] - kSqrt01_18 * (sh1[2, 1] * sh2[4, 2] + sh1[0, 1] * sh2[0, 2])
    sh3[4, 4] = sh1[1, 1] * sh2[3, 3] + kSqrt03_04 * sh1[2, 1] * sh2[2, 3] - kSqrt01_16 * (sh1[2, 1] * sh2[4, 3] + sh1[0, 1] * sh2[0, 3])
    sh3[4, 5] = kSqrt08_05 * sh1[1, 1] * sh2[3, 4] + kSqrt06_05 * sh1[2, 1] * sh2[2, 4] - kSqrt01_10 * (sh1[2, 1] * sh2[4, 4] + sh1[0, 1] * sh2[0, 4])
    sh3[4, 6] = kSqrt04_15 * (sh1[1, 2] * sh2[3, 4] - sh1[1, 0] * sh2[3, 0]) + kSqrt01_05 * (sh1[2, 2] * sh2[2, 4] - sh1[2, 0] * sh2[2, 0]) - kSqrt01_60 * ((sh1[2, 2] * sh2[4, 4] - sh1[2, 0] * sh2[4, 0]) + (sh1[0, 2] * sh2[0, 4] - sh1[0, 0] * sh2[0, 0]))

    sh3[5, 0] = kSqrt01_06 * (sh1[1, 2] * sh2[4, 0] + sh1[1, 0] * sh2[4, 4]) + kSqrt01_06 * ((sh1[2, 2] * sh2[3, 0] + sh1[2, 0] * sh2[3, 4]) - (sh1[0, 2] * sh2[1, 0] + sh1[0, 0] * sh2[1, 4]))
    sh3[5, 1] = sh1[1, 1] * sh2[4, 0] + (sh1[2, 1] * sh2[3, 0] - sh1[0, 1] * sh2[1, 0])
    sh3[5, 2] = kSqrt05_08 * sh1[1, 1] * sh2[4, 1] + kSqrt05_08 * (sh1[2, 1] * sh2[3, 1] - sh1[0, 1] * sh2[1, 1])
    sh3[5, 3] = kSqrt05_09 * sh1[1, 1] * sh2[4, 2] + kSqrt05_09 * (sh1[2, 1] * sh2[3, 2] - sh1[0, 1] * sh2[1, 2])
    sh3[5, 4] = kSqrt05_08 * sh1[1, 1] * sh2[4, 3] + kSqrt05_08 * (sh1[2, 1] * sh2[3, 3] - sh1[0, 1] * sh2[1, 3])
    sh3[5, 5] = sh1[1, 1] * sh2[4, 4] + (sh1[2, 1] * sh2[3, 4] - sh1[0, 1] * sh2[1, 4])
    sh3[5, 6] = kSqrt01_06 * (sh1[1, 2] * sh2[4, 4] - sh1[1, 0] * sh2[4, 0]) + kSqrt01_06 * ((sh1[2, 2] * sh2[3, 4] - sh1[2, 0] * sh2[3, 0]) - (sh1[0, 2] * sh2[1, 4] - sh1[0, 0] * sh2[1, 0]))

    sh3[6, 0] = kSqrt01_04 * ((sh1[2, 2] * sh2[4, 0] + sh1[2, 0] * sh2[4, 4]) - (sh1[0, 2] * sh2[0, 0] + sh1[0, 0] * sh2[0, 4]))
    sh3[6, 1] = kSqrt03_02 * (sh1[2, 1] * sh2[4, 0] - sh1[0, 1] * sh2[0, 0])
    sh3[6, 2] = kSqrt15_16 * (sh1[2, 1] * sh2[4, 1] - sh1[0, 1] * sh2[0, 1])
    sh3[6, 3] = kSqrt05_06 * (sh1[2, 1] * sh2[4, 2] - sh1[0, 1] * sh2[0, 2])
    sh3[6, 4] = kSqrt15_16 * (sh1[2, 1] * sh2[4, 3] - sh1[0, 1] * sh2[0, 3])
    sh3[6, 5] = kSqrt03_02 * (sh1[2, 1] * sh2[4, 4] - sh1[0, 1] * sh2[0, 4])
    sh3[6, 6] = kSqrt01_04 * ((sh1[2, 2] * sh2[4, 4] - sh1[2, 0] * sh2[4, 0]) - (sh1[0, 2] * sh2[0, 4] - sh1[0, 0] * sh2[0, 0]))

    return sh3

def get_sh4_torch(sh1, sh2, sh3):
    """
    Tính toán ma trận xoay SH bậc 4.
    Args:
        sh1 (torch.Tensor): Ma trận xoay SH bậc 1 (3, 3).
        sh2 (torch.Tensor): Ma trận xoay SH bậc 2 (5, 5).
        sh3 (torch.Tensor): Ma trận xoay SH bậc 3 (7, 7).
    Returns:
        torch.Tensor: Ma trận xoay SH bậc 4 (9, 9).
    """
    sh4 = torch.zeros((9, 9), dtype=sh1.dtype, device=sh1.device)

    sh4[0, 0] = kSqrt01_04 * ((sh1[2, 2] * sh3[0, 0] + sh1[2, 0] * sh3[0, 6]) + (sh1[0, 2] * sh3[6, 0] + sh1[0, 0] * sh3[6, 6]))
    sh4[0, 1] = kSqrt02_01 * (sh1[2, 1] * sh3[0, 0] + sh1[0, 1] * sh3[6, 0])
    sh4[0, 2] = kSqrt07_06 * (sh1[2, 1] * sh3[0, 1] + sh1[0, 1] * sh3[6, 1])
    sh4[0, 3] = kSqrt14_15 * (sh1[2, 1] * sh3[0, 2] + sh1[0, 1] * sh3[6, 2])
    sh4[0, 4] = kSqrt07_08 * (sh1[2, 1] * sh3[0, 3] + sh1[0, 1] * sh3[6, 3])
    sh4[0, 5] = kSqrt14_15 * (sh1[2, 1] * sh3[0, 4] + sh1[0, 1] * sh3[6, 4])
    sh4[0, 6] = kSqrt07_06 * (sh1[2, 1] * sh3[0, 5] + sh1[0, 1] * sh3[6, 5])
    sh4[0, 7] = kSqrt02_01 * (sh1[2, 1] * sh3[0, 6] + sh1[0, 1] * sh3[6, 6])
    sh4[0, 8] = kSqrt01_04 * ((sh1[2, 2] * sh3[0, 6] - sh1[2, 0] * sh3[0, 0]) + (sh1[0, 2] * sh3[6, 6] - sh1[0, 0] * sh3[6, 0]))

    sh4[1, 0] = kSqrt01_08 * (sh1[1, 2] * sh3[0, 0] + sh1[1, 0] * sh3[0, 6]) + kSqrt03_16 * ((sh1[2, 2] * sh3[1, 0] + sh1[2, 0] * sh3[1, 6]) + (sh1[0, 2] * sh3[5, 0] + sh1[0, 0] * sh3[5, 6]))
    sh4[1, 1] = sh1[1, 1] * sh3[0, 0] + kSqrt03_02 * (sh1[2, 1] * sh3[1, 0] + sh1[0, 1] * sh3[5, 0])
    sh4[1, 2] = kSqrt07_12 * sh1[1, 1] * sh3[0, 1] + kSqrt07_08 * (sh1[2, 1] * sh3[1, 1] + sh1[0, 1] * sh3[5, 1])
    sh4[1, 3] = kSqrt07_15 * sh1[1, 1] * sh3[0, 2] + kSqrt07_10 * (sh1[2, 1] * sh3[1, 2] + sh1[0, 1] * sh3[5, 2])
    sh4[1, 4] = kSqrt07_16 * sh1[1, 1] * sh3[0, 3] + kSqrt21_32 * (sh1[2, 1] * sh3[1, 3] + sh1[0, 1] * sh3[5, 3])
    sh4[1, 5] = kSqrt07_15 * sh1[1, 1] * sh3[0, 4] + kSqrt07_10 * (sh1[2, 1] * sh3[1, 4] + sh1[0, 1] * sh3[5, 4])
    sh4[1, 6] = kSqrt07_12 * sh1[1, 1] * sh3[0, 5] + kSqrt07_08 * (sh1[2, 1] * sh3[1, 5] + sh1[0, 1] * sh3[5, 5])
    sh4[1, 7] = sh1[1, 1] * sh3[0, 6] + kSqrt03_02 * (sh1[2, 1] * sh3[1, 6] + sh1[0, 1] * sh3[5, 6])
    sh4[1, 8] = kSqrt01_08 * (sh1[1, 2] * sh3[0, 6] - sh1[1, 0] * sh3[0, 0]) + kSqrt03_16 * ((sh1[2, 2] * sh3[1, 6] - sh1[2, 0] * sh3[1, 0]) + (sh1[0, 2] * sh3[5, 6] - sh1[0, 0] * sh3[5, 0]))

    sh4[2, 0] = kSqrt03_14 * (sh1[1, 2] * sh3[1, 0] + sh1[1, 0] * sh3[1, 6]) + kSqrt15_112 * ((sh1[2, 2] * sh3[2, 0] + sh1[2, 0] * sh3[2, 6]) + (sh1[0, 2] * sh3[4, 0] + sh1[0, 0] * sh3[4, 6])) - kSqrt01_112 * ((sh1[2, 2] * sh3[0, 0] + sh1[2, 0] * sh3[0, 6]) - (sh1[0, 2] * sh3[6, 0] + sh1[0, 0] * sh3[6, 6]))
    sh4[2, 1] = kSqrt12_07 * sh1[1, 1] * sh3[1, 0] + kSqrt15_14 * (sh1[2, 1] * sh3[2, 0] + sh1[0, 1] * sh3[4, 0]) - kSqrt01_14 * (sh1[2, 1] * sh3[0, 0] - sh1[0, 1] * sh3[6, 0])
    sh4[2, 2] = sh1[1, 1] * sh3[1, 1] + kSqrt05_08 * (sh1[2, 1] * sh3[2, 1] + sh1[0, 1] * sh3[4, 1]) - kSqrt01_24 * (sh1[2, 1] * sh3[0, 1] - sh1[0, 1] * sh3[6, 1])
    sh4[2, 3] = kSqrt04_05 * sh1[1, 1] * sh3[1, 2] + kSqrt01_02 * (sh1[2, 1] * sh3[2, 2] + sh1[0, 1] * sh3[4, 2]) - kSqrt01_30 * (sh1[2, 1] * sh3[0, 2] - sh1[0, 1] * sh3[6, 2])
    sh4[2, 4] = kSqrt03_04 * sh1[1, 1] * sh3[1, 3] + kSqrt15_32 * (sh1[2, 1] * sh3[2, 3] + sh1[0, 1] * sh3[4, 3]) - kSqrt01_32 * (sh1[2, 1] * sh3[0, 3] - sh1[0, 1] * sh3[6, 3])
    sh4[2, 5] = kSqrt04_05 * sh1[1, 1] * sh3[1, 4] + kSqrt01_02 * (sh1[2, 1] * sh3[2, 4] + sh1[0, 1] * sh3[4, 4]) - kSqrt01_30 * (sh1[2, 1] * sh3[0, 4] - sh1[0, 1] * sh3[6, 4])
    sh4[2, 6] = sh1[1, 1] * sh3[1, 5] + kSqrt05_08 * (sh1[2, 1] * sh3[2, 5] + sh1[0, 1] * sh3[4, 5]) - kSqrt01_24 * (sh1[2, 1] * sh3[0, 5] - sh1[0, 1] * sh3[6, 5])
    sh4[2, 7] = kSqrt12_07 * sh1[1, 1] * sh3[1, 6] + kSqrt15_14 * (sh1[2, 1] * sh3[2, 6] + sh1[0, 1] * sh3[4, 6]) - kSqrt01_14 * (sh1[2, 1] * sh3[0, 6] - sh1[0, 1] * sh3[6, 6])
    sh4[2, 8] = kSqrt03_14 * (sh1[1, 2] * sh3[1, 6] - sh1[1, 0] * sh3[1, 0]) + kSqrt15_112 * ((sh1[2, 2] * sh3[2, 6] - sh1[2, 0] * sh3[2, 0]) + (sh1[0, 2] * sh3[4, 6] - sh1[0, 0] * sh3[4, 0])) - kSqrt01_112 * ((sh1[2, 2] * sh3[0, 6] - sh1[2, 0] * sh3[0, 0]) - (sh1[0, 2] * sh3[6, 6] - sh1[0, 0] * sh3[6, 0]))

    sh4[3, 0] = kSqrt15_56 * (sh1[1, 2] * sh3[2, 0] + sh1[1, 0] * sh3[2, 6]) + kSqrt05_28 * (sh1[0, 2] * sh3[3, 0] + sh1[0, 0] * sh3[3, 6]) - kSqrt03_112 * ((sh1[2, 2] * sh3[1, 0] + sh1[2, 0] * sh3[1, 6]) - (sh1[0, 2] * sh3[5, 0] + sh1[0, 0] * sh3[5, 6]))
    sh4[3, 1] = kSqrt15_07 * sh1[1, 1] * sh3[2, 0] + kSqrt10_07 * sh1[0, 1] * sh3[3, 0] - kSqrt03_14 * (sh1[2, 1] * sh3[1, 0] - sh1[0, 1] * sh3[5, 0])
    sh4[3, 2] = kSqrt05_04 * sh1[1, 1] * sh3[2, 1] + kSqrt05_06 * sh1[0, 1] * sh3[3, 1] - kSqrt01_08 * (sh1[2, 1] * sh3[1, 1] - sh1[0, 1] * sh3[5, 1])
    sh4[3, 3] = sh1[1, 1] * sh3[2, 2] + kSqrt02_03 * sh1[0, 1] * sh3[3, 2] - kSqrt01_10 * (sh1[2, 1] * sh3[1, 2] - sh1[0, 1] * sh3[5, 2])
    sh4[3, 4] = kSqrt15_16 * sh1[1, 1] * sh3[2, 3] + kSqrt05_08 * sh1[0, 1] * sh3[3, 3] - kSqrt03_32 * (sh1[2, 1] * sh3[1, 3] - sh1[0, 1] * sh3[5, 3])
    sh4[3, 5] = sh1[1, 1] * sh3[2, 4] + kSqrt02_03 * sh1[0, 1] * sh3[3, 4] - kSqrt01_10 * (sh1[2, 1] * sh3[1, 4] - sh1[0, 1] * sh3[5, 4])
    sh4[3, 6] = kSqrt05_04 * sh1[1, 1] * sh3[2, 5] + kSqrt05_06 * sh1[0, 1] * sh3[3, 5] - kSqrt01_08 * (sh1[2, 1] * sh3[1, 5] - sh1[0, 1] * sh3[5, 5])
    sh4[3, 7] = kSqrt15_07 * sh1[1, 1] * sh3[2, 6] + kSqrt10_07 * sh1[0, 1] * sh3[3, 6] - kSqrt03_14 * (sh1[2, 1] * sh3[1, 6] - sh1[0, 1] * sh3[5, 6])
    sh4[3, 8] = kSqrt15_56 * (sh1[1, 2] * sh3[2, 6] - sh1[1, 0] * sh3[2, 0]) + kSqrt05_28 * (sh1[0, 2] * sh3[3, 6] - sh1[0, 0] * sh3[3, 0]) - kSqrt03_112 * ((sh1[2, 2] * sh3[1, 6] - sh1[2, 0] * sh3[1, 0]) - (sh1[0, 2] * sh3[5, 6] - sh1[0, 0] * sh3[5, 0]))

    sh4[4, 0] = kSqrt02_07 * (sh1[1, 2] * sh3[3, 0] + sh1[1, 0] * sh3[3, 6]) - kSqrt03_28 * ((sh1[2, 2] * sh3[4, 0] + sh1[2, 0] * sh3[4, 6]) + (sh1[0, 2] * sh3[2, 0] + sh1[0, 0] * sh3[2, 6]))
    sh4[4, 1] = kSqrt16_07 * sh1[1, 1] * sh3[3, 0] - kSqrt06_07 * (sh1[2, 1] * sh3[4, 0] + sh1[0, 1] * sh3[2, 0])
    sh4[4, 2] = kSqrt04_03 * sh1[1, 1] * sh3[3, 1] - kSqrt01_02 * (sh1[2, 1] * sh3[4, 1] + sh1[0, 1] * sh3[2, 1])
    sh4[4, 3] = kSqrt16_15 * sh1[1, 1] * sh3[3, 2] - kSqrt02_05 * (sh1[2, 1] * sh3[4, 2] + sh1[0, 1] * sh3[2, 2])
    sh4[4, 4] = sh1[1, 1] * sh3[3, 3] - kSqrt03_08 * (sh1[2, 1] * sh3[4, 3] + sh1[0, 1] * sh3[2, 3])
    sh4[4, 5] = kSqrt16_15 * sh1[1, 1] * sh3[3, 4] - kSqrt02_05 * (sh1[2, 1] * sh3[4, 4] + sh1[0, 1] * sh3[2, 4])
    sh4[4, 6] = kSqrt04_03 * sh1[1, 1] * sh3[3, 5] - kSqrt01_02 * (sh1[2, 1] * sh3[4, 5] + sh1[0, 1] * sh3[2, 5])
    sh4[4, 7] = kSqrt16_07 * sh1[1, 1] * sh3[3, 6] - kSqrt06_07 * (sh1[2, 1] * sh3[4, 6] + sh1[0, 1] * sh3[2, 6])
    sh4[4, 8] = kSqrt02_07 * (sh1[1, 2] * sh3[3, 6] - sh1[1, 0] * sh3[3, 0]) - kSqrt03_28 * ((sh1[2, 2] * sh3[4, 6] - sh1[2, 0] * sh3[4, 0]) + (sh1[0, 2] * sh3[2, 6] - sh1[0, 0] * sh3[2, 0]))

    sh4[5, 0] = kSqrt15_56 * (sh1[1, 2] * sh3[4, 0] + sh1[1, 0] * sh3[4, 6]) + kSqrt05_28 * (sh1[2, 2] * sh3[3, 0] + sh1[2, 0] * sh3[3, 6]) - kSqrt03_112 * ((sh1[2, 2] * sh3[5, 0] + sh1[2, 0] * sh3[5, 6]) + (sh1[0, 2] * sh3[1, 0] + sh1[0, 0] * sh3[1, 6]))
    sh4[5, 1] = kSqrt15_07 * sh1[1, 1] * sh3[4, 0] + kSqrt10_07 * sh1[2, 1] * sh3[3, 0] - kSqrt03_14 * (sh1[2, 1] * sh3[5, 0] + sh1[0, 1] * sh3[1, 0])
    sh4[5, 2] = kSqrt05_04 * sh1[1, 1] * sh3[4, 1] + kSqrt05_06 * sh1[2, 1] * sh3[3, 1] - kSqrt01_08 * (sh1[2, 1] * sh3[5, 1] + sh1[0, 1] * sh3[1, 1])
    sh4[5, 3] = sh1[1, 1] * sh3[4, 2] + kSqrt02_03 * sh1[2, 1] * sh3[3, 2] - kSqrt01_10 * (sh1[2, 1] * sh3[5, 2] + sh1[0, 1] * sh3[1, 2])
    sh4[5, 4] = kSqrt15_16 * sh1[1, 1] * sh3[4, 3] + kSqrt05_08 * sh1[2, 1] * sh3[3, 3] - kSqrt03_32 * (sh1[2, 1] * sh3[5, 3] + sh1[0, 1] * sh3[1, 3])
    sh4[5, 5] = sh1[1, 1] * sh3[4, 4] + kSqrt02_03 * sh1[2, 1] * sh3[3, 4] - kSqrt01_10 * (sh1[2, 1] * sh3[5, 4] + sh1[0, 1] * sh3[1, 4])
    sh4[5, 6] = kSqrt05_04 * sh1[1, 1] * sh3[4, 5] + kSqrt05_06 * sh1[2, 1] * sh3[3, 5] - kSqrt01_08 * (sh1[2, 1] * sh3[5, 5] + sh1[0, 1] * sh3[1, 5])
    sh4[5, 7] = kSqrt15_07 * sh1[1, 1] * sh3[4, 6] + kSqrt10_07 * sh1[2, 1] * sh3[3, 6] - kSqrt03_14 * (sh1[2, 1] * sh3[5, 6] + sh1[0, 1] * sh3[1, 6])
    sh4[5, 8] = kSqrt15_56 * (sh1[1, 2] * sh3[4, 6] - sh1[1, 0] * sh3[4, 0]) + kSqrt05_28 * (sh1[2, 2] * sh3[3, 6] - sh1[2, 0] * sh3[3, 0]) - kSqrt03_112 * ((sh1[2, 2] * sh3[5, 6] - sh1[2, 0] * sh3[5, 0]) + (sh1[0, 2] * sh3[1, 6] - sh1[0, 0] * sh3[1, 0]))

    sh4[6, 0] = kSqrt03_14 * (sh1[1, 2] * sh3[5, 0] + sh1[1, 0] * sh3[5, 6]) + kSqrt15_112 * ((sh1[2, 2] * sh3[4, 0] + sh1[2, 0] * sh3[4, 6]) - (sh1[0, 2] * sh3[2, 0] + sh1[0, 0] * sh3[2, 6])) - kSqrt01_112 * ((sh1[2, 2] * sh3[6, 0] + sh1[2, 0] * sh3[6, 6]) + (sh1[0, 2] * sh3[0, 0] + sh1[0, 0] * sh3[0, 6]))
    sh4[6, 1] = kSqrt12_07 * sh1[1, 1] * sh3[5, 0] + kSqrt15_14 * (sh1[2, 1] * sh3[4, 0] - sh1[0, 1] * sh3[2, 0]) - kSqrt01_14 * (sh1[2, 1] * sh3[6, 0] + sh1[0, 1] * sh3[0, 0])
    sh4[6, 2] = sh1[1, 1] * sh3[5, 1] + kSqrt05_08 * (sh1[2, 1] * sh3[4, 1] - sh1[0, 1] * sh3[2, 1]) - kSqrt01_24 * (sh1[2, 1] * sh3[6, 1] + sh1[0, 1] * sh3[0, 1])
    sh4[6, 3] = kSqrt04_05 * sh1[1, 1] * sh3[5, 2] + kSqrt01_02 * (sh1[2, 1] * sh3[4, 2] - sh1[0, 1] * sh3[2, 2]) - kSqrt01_30 * (sh1[2, 1] * sh3[6, 2] + sh1[0, 1] * sh3[0, 2])
    sh4[6, 4] = kSqrt03_04 * sh1[1, 1] * sh3[5, 3] + kSqrt15_32 * (sh1[2, 1] * sh3[4, 3] - sh1[0, 1] * sh3[2, 3]) - kSqrt01_32 * (sh1[2, 1] * sh3[6, 3] + sh1[0, 1] * sh3[0, 3])
    sh4[6, 5] = kSqrt04_05 * sh1[1, 1] * sh3[5, 4] + kSqrt01_02 * (sh1[2, 1] * sh3[4, 4] - sh1[0, 1] * sh3[2, 4]) - kSqrt01_30 * (sh1[2, 1] * sh3[6, 4] + sh1[0, 1] * sh3[0, 4])
    sh4[6, 6] = sh1[1, 1] * sh3[5, 5] + kSqrt05_08 * (sh1[2, 1] * sh3[4, 5] - sh1[0, 1] * sh3[2, 5]) - kSqrt01_24 * (sh1[2, 1] * sh3[6, 5] + sh1[0, 1] * sh3[0, 5])
    sh4[6, 7] = kSqrt12_07 * sh1[1, 1] * sh3[5, 6] + kSqrt15_14 * (sh1[2, 1] * sh3[4, 6] - sh1[0, 1] * sh3[2, 6]) - kSqrt01_14 * (sh1[2, 1] * sh3[6, 6] + sh1[0, 1] * sh3[0, 6])
    sh4[6, 8] = kSqrt03_14 * (sh1[1, 2] * sh3[5, 6] - sh1[1, 0] * sh3[5, 0]) + kSqrt15_112 * ((sh1[2, 2] * sh3[4, 6] - sh1[2, 0] * sh3[4, 0]) - (sh1[0, 2] * sh3[2, 6] - sh1[0, 0] * sh3[2, 0])) - kSqrt01_112 * ((sh1[2, 2] * sh3[6, 6] - sh1[2, 0] * sh3[6, 0]) + (sh1[0, 2] * sh3[0, 6] - sh1[0, 0] * sh3[0, 0]))

    sh4[7, 0] = kSqrt01_08 * (sh1[1, 2] * sh3[6, 0] + sh1[1, 0] * sh3[6, 6]) + kSqrt03_16 * ((sh1[2, 2] * sh3[5, 0] + sh1[2, 0] * sh3[5, 6]) - (sh1[0, 2] * sh3[1, 0] + sh1[0, 0] * sh3[1, 6]))
    sh4[7, 1] = sh1[1, 1] * sh3[6, 0] + kSqrt03_02 * (sh1[2, 1] * sh3[5, 0] - sh1[0, 1] * sh3[1, 0])
    sh4[7, 2] = kSqrt07_12 * sh1[1, 1] * sh3[6, 1] + kSqrt07_08 * (sh1[2, 1] * sh3[5, 1] - sh1[0, 1] * sh3[1, 1])
    sh4[7, 3] = kSqrt07_15 * sh1[1, 1] * sh3[6, 2] + kSqrt07_10 * (sh1[2, 1] * sh3[5, 2] - sh1[0, 1] * sh3[1, 2])
    sh4[7, 4] = kSqrt07_16 * sh1[1, 1] * sh3[6, 3] + kSqrt21_32 * (sh1[2, 1] * sh3[5, 3] - sh1[0, 1] * sh3[1, 3])
    sh4[7, 5] = kSqrt07_15 * sh1[1, 1] * sh3[6, 4] + kSqrt07_10 * (sh1[2, 1] * sh3[5, 4] - sh1[0, 1] * sh3[1, 4])
    sh4[7, 6] = kSqrt07_12 * sh1[1, 1] * sh3[6, 5] + kSqrt07_08 * (sh1[2, 1] * sh3[5, 5] - sh1[0, 1] * sh3[1, 5])
    sh4[7, 7] = sh1[1, 1] * sh3[6, 6] + kSqrt03_02 * (sh1[2, 1] * sh3[5, 6] - sh1[0, 1] * sh3[1, 6])
    sh4[7, 8] = kSqrt01_08 * (sh1[1, 2] * sh3[6, 6] - sh1[1, 0] * sh3[6, 0]) + kSqrt03_16 * ((sh1[2, 2] * sh3[5, 6] - sh1[2, 0] * sh3[5, 0]) - (sh1[0, 2] * sh3[1, 6] - sh1[0, 0] * sh3[1, 0]))

    sh4[8, 0] = kSqrt01_04 * ((sh1[2, 2] * sh3[6, 0] + sh1[2, 0] * sh3[6, 6]) - (sh1[0, 2] * sh3[0, 0] + sh1[0, 0] * sh3[0, 6]))
    sh4[8, 1] = kSqrt02_01 * (sh1[2, 1] * sh3[6, 0] - sh1[0, 1] * sh3[0, 0])
    sh4[8, 2] = kSqrt07_06 * (sh1[2, 1] * sh3[6, 1] - sh1[0, 1] * sh3[0, 1])
    sh4[8, 3] = kSqrt14_15 * (sh1[2, 1] * sh3[6, 2] - sh1[0, 1] * sh3[0, 2])
    sh4[8, 4] = kSqrt07_08 * (sh1[2, 1] * sh3[6, 3] - sh1[0, 1] * sh3[0, 3])
    sh4[8, 5] = kSqrt14_15 * (sh1[2, 1] * sh3[6, 4] - sh1[0, 1] * sh3[0, 4])
    sh4[8, 6] = kSqrt07_06 * (sh1[2, 1] * sh3[6, 5] - sh1[0, 1] * sh3[0, 5])
    sh4[8, 7] = kSqrt02_01 * (sh1[2, 1] * sh3[6, 6] - sh1[0, 1] * sh3[0, 6])
    sh4[8, 8] = kSqrt01_04 * ((sh1[2, 2] * sh3[6, 6] - sh1[2, 0] * sh3[6, 0]) - (sh1[0, 2] * sh3[0, 6] - sh1[0, 0] * sh3[0, 0]))

    return sh4


class SHRotatorTorch:
    def __init__(self, R, deg=3):
        """
        Khởi tạo bộ xoay SH sử dụng PyTorch.
        Args:
            R (torch.Tensor): Ma trận xoay (3, 3). Phải là tensor PyTorch.
            deg (int): Bậc SH cao nhất cần tính toán (tối đa là 4).
        """
        if not isinstance(R, torch.Tensor):
            raise TypeError("Input R must be a torch.Tensor")

        self.deg = deg
        self.device = R.device
        self.dtype = R.dtype

        # Đảm bảo các hằng số cùng device và dtype với R nếu chúng chưa được đặt
        global _device, _dtype
        if _device is None:
            _device = self.device
            _dtype = self.dtype
            # Có thể cần cập nhật lại các hằng số nếu device/dtype thay đổi
            # Hoặc tốt hơn là truyền device/dtype vào khi tạo hằng số ngay từ đầu
            # hoặc .to(device=self.device, dtype=self.dtype) trong các hàm get_shX

        if deg > 0:
            self.sh1 = get_sh1_torch(R)
            self.sh1_T = self.sh1.T # Tính trước ma trận chuyển vị
        if deg > 1:
            self.sh2 = get_sh2_torch(self.sh1)
            self.sh2_T = self.sh2.T
        if deg > 2:
            self.sh3 = get_sh3_torch(self.sh1, self.sh2)
            self.sh3_T = self.sh3.T
        if deg > 3:
            self.sh4 = get_sh4_torch(self.sh1, self.sh2, self.sh3)
            self.sh4_T = self.sh4.T
        if deg > 4:
            raise NotImplementedError("SH rotation up to degree 4 is supported.")

    def __call__(self, shs_in):
        """
        Áp dụng phép xoay SH lên các hệ số SH đầu vào.
        Args:
            shs_in (torch.Tensor): Tensor chứa các hệ số SH.
                                   Shape có thể là (num_coeffs) hoặc (batch_size, num_coeffs).
                                   num_coeffs phải tương ứng với bậc `deg`.
                                   Ví dụ: deg=0 -> 1, deg=1 -> 4, deg=2 -> 9, deg=3 -> 16, deg=4 -> 25.
        Returns:
            torch.Tensor: Tensor chứa các hệ số SH đã được xoay, cùng shape với shs_in.
        """
        if not isinstance(shs_in, torch.Tensor):
            raise TypeError("Input shs_in must be a torch.Tensor")
        if shs_in.device != self.device or shs_in.dtype != self.dtype:
             shs_in = shs_in.to(device=self.device, dtype=self.dtype)

        expected_coeffs = (self.deg + 1)**2
        if shs_in.shape[-1] != expected_coeffs:
            raise ValueError(f"Input shs_in has {shs_in.shape[-1]} coefficients, "
                             f"but expected {expected_coeffs} for degree {self.deg}")

        shs_out_parts = []
        # deg 0 (l=0, m=0) - không thay đổi khi xoay
        shs_out_parts.append(shs_in[..., 0:1])

        # deg 1 (l=1, m=-1,0,1) - 3 hệ số
        if self.deg > 0:
            # Phép nhân ma trận: (B, 3) @ (3, 3) -> (B, 3)
            # Hoặc (3) @ (3, 3) -> (3) nếu không có batch dim
            shs_deg1 = shs_in[..., 1:4]
            shs_out_parts.append(shs_deg1 @ self.sh1_T)

        # deg 2 (l=2, m=-2,...,2) - 5 hệ số
        if self.deg > 1:
            shs_deg2 = shs_in[..., 4:9]
            shs_out_parts.append(shs_deg2 @ self.sh2_T)

        # deg 3 (l=3, m=-3,...,3) - 7 hệ số
        if self.deg > 2:
            shs_deg3 = shs_in[..., 9:16]
            shs_out_parts.append(shs_deg3 @ self.sh3_T)

        # deg 4 (l=4, m=-4,...,4) - 9 hệ số
        if self.deg > 3:
            shs_deg4 = shs_in[..., 16:25]
            shs_out_parts.append(shs_deg4 @ self.sh4_T)

        # Nối các phần lại theo chiều cuối cùng
        shs_out = torch.cat(shs_out_parts, dim=-1)

        # Kiểm tra shape cuối cùng
        assert shs_out.shape == shs_in.shape, f"Output shape {shs_out.shape} mismatch input shape {shs_in.shape}"
        return shs_out

# # --- Ví dụ sử dụng ---
# if __name__ == '__main__':
#     # 1. Tạo ma trận xoay ngẫu nhiên (ví dụ)
#     # Sử dụng scipy để tạo ma trận xoay hợp lệ cho dễ
#     try:
#         from scipy.spatial.transform import Rotation as R_scipy
#         r_quat = R_scipy.random().as_matrix()
#         R_np = np.array(r_quat, dtype=np.float32)
#     except ImportError:
#         print("scipy not found, using a simple identity matrix for R.")
#         R_np = np.eye(3, dtype=np.float32)

#     R_torch = torch.from_numpy(R_np) #.cuda() # Chuyển sang GPU nếu muốn

#     # 2. Tạo hệ số SH đầu vào ngẫu nhiên (ví dụ cho deg=3)
#     deg = 3
#     num_coeffs = (deg + 1)**2 # 1 + 3 + 5 + 7 = 16
#     batch_size = 2 # Ví dụ với batch
#     shs_in_np = np.random.rand(batch_size, num_coeffs).astype(np.float32)
#     shs_in_torch = torch.from_numpy(shs_in_np) #.cuda() # Chuyển sang GPU nếu muốn

#     # 3. Sử dụng phiên bản NumPy gốc
#     print("--- NumPy Version ---")
#     rotator_np = SHRotator(R_np, deg=deg)
#     shs_out_np = rotator_np(shs_in_np)
#     print("Input SHs (NumPy):\n", shs_in_np)
#     print("Output SHs (NumPy):\n", shs_out_np)

#     # 4. Sử dụng phiên bản PyTorch
#     print("\n--- PyTorch Version ---")
#     # Đặt device và dtype cho hằng số nếu chưa làm
#     _device = R_torch.device
#     _dtype = R_torch.dtype
#     # (Trong code thực tế, nên xử lý việc này tốt hơn, ví dụ truyền device/dtype vào __init__)

#     rotator_torch = SHRotatorTorch(R_torch, deg=deg)
#     shs_out_torch = rotator_torch(shs_in_torch)
#     print("Input SHs (Torch):\n", shs_in_torch)
#     print("Output SHs (Torch):\n", shs_out_torch)

#     # 5. So sánh kết quả
#     print("\n--- Comparison ---")
#     # Chuyển kết quả PyTorch về CPU và NumPy để so sánh
#     shs_out_torch_np = shs_out_torch.cpu().numpy()
#     diff = np.abs(shs_out_np - shs_out_torch_np)
#     print("Max absolute difference:", np.max(diff))
#     print("Are results close?", np.allclose(shs_out_np, shs_out_torch_np, atol=1e-6)) # Tăng atol nếu cần

#     # Kiểm tra tốc độ (ví dụ đơn giản)
#     import time
#     n_runs = 100

#     start_np = time.time()
#     for _ in range(n_runs):
#         _ = rotator_np(shs_in_np)
#     end_np = time.time()
#     print(f"\nNumPy time ({n_runs} runs): {(end_np - start_np):.6f} s")

#     # Đảm bảo tensor trên đúng device trước khi đo thời gian GPU
#     shs_in_torch_dev = shs_in_torch.to(rotator_torch.device)
#     if R_torch.device.type == 'cuda':
#          torch.cuda.synchronize() # Đồng bộ trước khi đo GPU
#     start_torch = time.time()
#     for _ in range(n_runs):
#         _ = rotator_torch(shs_in_torch_dev)
#     if R_torch.device.type == 'cuda':
#         torch.cuda.synchronize() # Đồng bộ sau khi đo GPU
#     end_torch = time.time()
#     print(f"PyTorch time ({R_torch.device}, {n_runs} runs): {(end_torch - start_torch):.6f} s")