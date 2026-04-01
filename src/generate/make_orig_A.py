"""Generate a uniform orig_A matrix for the preprocessing pipeline.

이 스크립트는 LWE (Learning With Errors) 문제를 위한 행렬 A를 생성합니다.
생성된 행렬은 다음과 같이 사용할 수 있습니다:

    python3 src/generate/preprocess.py --reload_data <output_path> ...
    python3 src/generate/generate_A_b.py --orig_A_path <output_path> ...

기본적으로 균일 분포(uniform distribution)에서 난수를 생성하여 
지정된 차원과 modulus에 따른 LWE 행렬을 만듭니다.
"""

import argparse  # 명령줄 인자 파싱을 위한 라이브러리
import math  # 수학 함수 (log2 등)
import os  # 파일 경로 조작

import numpy as np  # 수치 계산 라이브러리


def get_parser():
    """명령줄 인자를 파싱하기 위한 ArgumentParser를 생성합니다.
    
    반환값:
        argparse.ArgumentParser: 설정된 파서 객체
    
    지원하는 인자들:
        - N: LWE 문제의 차원
        - Q: LWE 모듈러스 (모듈로 연산의 기수)
        - num_rows: 생성할 LWE 행의 개수
        - row_multiplier: num_rows를 계산하기 위한 승수 (미지정 시)
        - seed: 난수 생성 재현성을 위한 시드값
        - representation: 행렬 원소의 표현 형식 (mod_q 또는 centered)
        - dump_path: 출력 디렉토리 경로
        - output_path: 생성된 .npy 파일의 전체 경로
    """
    parser = argparse.ArgumentParser(
        description="Generate an orig_A.npy file for preprocess.py."
    )
    # LWE 문제의 차원 설정
    parser.add_argument("--N", type=int, required=True, help="LWE dimension.")
    # LWE 모듈러스 (일반적으로 소수 또는 2의 거듭제곱)
    parser.add_argument("--Q", type=int, required=True, help="LWE modulus.")
    # 생성할 행렬의 행 개수 (음수면 row_multiplier * N으로 계산)
    parser.add_argument(
        "--num_rows",
        type=int,
        default=-1,
        help="Number of LWE rows to generate. Defaults to row_multiplier * N.",
    )
    # row_multiplier: num_rows를 명시하지 않았을 때 사용
    # 기본값 4는 보통 4*N개의 방정식이 필요하다는 휴리스틱에서 옴
    parser.add_argument(
        "--row_multiplier",
        type=int,
        default=4,
        help="Used when num_rows is omitted. Default generates 4 * N rows.",
    )
    # 난수 생성기의 시드 (재현성 보장)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for reproducible generation.",
    )
    parser.add_argument(
        "--bit_generator",
        choices=["pcg64dxsm", "pcg64", "philox", "mt19937"],
        default="pcg64dxsm",
        help=(
            "Explicit NumPy bit generator for orig_A. "
            "Defaults to PCG64DXSM to avoid ambiguity about RNG provenance."
        ),
    )
    # 행렬 원소의 값 범위 표현 방식
    # mod_q: [0, Q) 범위
    # centered: 중심화된 범위 (음수 포함)
    parser.add_argument(
        "--representation",
        choices=["mod_q", "centered"],
        default="mod_q",
        help="Whether to save entries in [0, Q) or centered form.",
    )
    # 기본 출력 디렉토리 (output_path가 지정되지 않았을 때 사용)
    parser.add_argument(
        "--dump_path",
        type=str,
        default="data/generated_orig_A",
        help="Output directory when output_path is not provided.",
    )
    # 생성된 파일의 전체 경로 (이 값이 지정되면 dump_path는 무시됨)
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Full path for the generated .npy file.",
    )
    return parser


def make_rng(seed, bit_generator):
    """Return an explicit NumPy Generator for reproducible, exact-uniform sampling."""
    seed_sequence = np.random.SeedSequence(seed)
    bit_generators = {
        "pcg64dxsm": np.random.PCG64DXSM,
        "pcg64": np.random.PCG64,
        "philox": np.random.Philox,
        "mt19937": np.random.MT19937,
    }
    return np.random.Generator(bit_generators[bit_generator](seed_sequence))


def get_output_path(args):
    """출력 파일의 경로를 결정합니다.
    
    매개변수:
        args: ArgumentParser로 파싱된 명령줄 인자 객체
    
    반환값:
        str: 생성할 .npy 파일의 전체 경로
    
    동작:
        - args.output_path가 지정되면 그 값을 반환
        - 미지정 이면 args.dump_path 디렉토리에 
          파라미터들을 포함한 이름으로 파일명 생성
        - 파일명 형식: origA_n{N}_logq{logq}.npy
    """
    # 사용자가 직접 output_path를 지정했으면 그것을 사용
    if args.output_path:
        return args.output_path

    # log_q 계산: Q의 binary representation에 필요한 비트 개수
    # 예: Q=256 -> log_q=8, Q=512 -> log_q=9
    logq = int(math.ceil(math.log2(args.Q)))
    
    # repo 예제 스타일과 맞춘 파일명 생성
    filename = f"origA_n{args.N}_logq{logq}.npy"
    
    # dump_path 디렉토리에 파일명을 결합하여 전체 경로 생성
    return os.path.join(args.dump_path, filename)


def sample_orig_a(num_rows, n, q, representation, rng):
    """균일 분포에서 LWE 행렬 A를 생성합니다.
    
    매개변수:
        num_rows (int): 생성할 행의 개수 (LWE 샘플의 개수)
        n (int): 각 행의 열의 개수 (LWE 차원)
        q (int): 모듈러스 (모든 원소는 mod q로 계산)
        representation (str): 값의 표현 형식
            - "mod_q": [0, q) 범위의 음이 아닌 정수
            - "centered": [-q/2, q/2] 범위의 중심화된 정수
        rng (np.random.Generator): NumPy 난수 생성기
    
    반환값:
        np.ndarray: 형태 (num_rows, n), dtype=int64인 행렬
                   각 원소는 균일 분포에서 샘플링
    
    설명:
        - "mod_q"는 전통적인 LWE 정의에 따름
        - "centered"는 일부 암호화 구현에서 흔하게 사용됨
          (센트럼 근처에 값들이 분포하므로 분석에 유리)
        - 둘 다 rng.integers()를 사용하므로 modulo reduction 없이
          정확한 discrete uniform 분포를 샘플링합니다.
    """
    if representation == "mod_q":
        # [0, q) 범위에서 균일 분포로 샘플링
        return rng.integers(0, q, size=(num_rows, n), dtype=np.int64)

    # centered 표현: [-q/2, q/2] 범위
    # q가 홀수면 상한이 (q//2 + 1), 짝수면 q//2
    low = -(q // 2)
    high = q // 2 + 1 if q % 2 else q // 2
    return rng.integers(low, high, size=(num_rows, n), dtype=np.int64)


def main(args):
    """프로그램의 메인 함수입니다.
    
    매개변수:
        args: 파싱된 명령줄 인자 객체
    
    동작 순서:
        1. 입력 파라미터 유효성 검증
        2. num_rows 값이 음수면 자동 계산
        3. 출력 디렉토리 생성
        4. 난수 생성기 초기화
        5. LWE 행렬 생성
        6. NumPy .npy 형식으로 저장
        7. 생성된 행렬의 정보 출력
    """
    
    # === 입력 파라미터 유효성 검증 ===
    
    # N(차원)은 양수여야 함
    if args.N <= 0:
        raise ValueError(f"N must be positive, got {args.N}")
    
    # Q(모듈러스)는 최소한 2 이상이어야 함
    if args.Q <= 1:
        raise ValueError(f"Q must be at least 2, got {args.Q}")
    
    # row_multiplier는 양수여야 함
    if args.row_multiplier <= 0:
        raise ValueError(
            f"row_multiplier must be positive, got {args.row_multiplier}"
        )

    # === num_rows 자동 계산 (음수일 경우) ===
    if args.num_rows < 0:
        args.num_rows = args.row_multiplier * args.N
    
    # 최종 계산된 num_rows도 양수여야 함
    if args.num_rows <= 0:
        raise ValueError(f"num_rows must be positive, got {args.num_rows}")

    # === 출력 경로 결정 및 디렉토리 생성 ===
    output_path = get_output_path(args)
    # 출력 디렉토리가 없으면 생성 (exist_ok=True로 기존 디렉토리는 무시)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # === 난수 생성기 초기화 ===
    # seed를 설정하면 같은 파라미터로 재실행했을 때 동일한 행렬 생성 가능
    rng = make_rng(args.seed, args.bit_generator)
    
    # === LWE 행렬 생성 ===
    orig_a = sample_orig_a(
        args.num_rows, args.N, args.Q, args.representation, rng
    )
    
    # === NumPy 바이너리 형식으로 저장 ===
    np.save(output_path, orig_a)

    # === 생성 결과 정보 출력 ===
    print(f"Saved orig_A to {output_path}")
    print(f"shape={orig_a.shape} dtype={orig_a.dtype}")
    print(f"min={orig_a.min()} max={orig_a.max()}")
    print(f"bit_generator={args.bit_generator} seed={args.seed}")
    print(
        "Use this file with "
        f"--reload_data {output_path} and --orig_A_path {output_path}"
    )


if __name__ == "__main__":
    # 스크립트가 직접 실행된 경우에만 실행
    # (다른 모듈에서 import 되었을 때는 실행되지 않음)
    # 명령줄 인자를 파싱하고 main() 함수 호출
    main(get_parser().parse_args())
