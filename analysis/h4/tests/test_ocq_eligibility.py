import pandas as pd

from h4.survey.eligibility import eligible_long_rows, mask_ineligible_ocq, ocq_nonconsumer


def test_ocq_only_exclusion_preserves_other_scales_and_stable_ids():
    wide = pd.DataFrame({
        "set_id": ["C"] * 4,
        "C08_chocolate_frequency": ["6", "6.0", "5", None],
        "C08_item01_human": [1, 2, 3, 4],
        "C08_wording_ai": [5, 4, 3, 2],
        "C08_preference": [1, 2, 1, 2],
        "C07_item01_human": [1, 2, 3, 4],
    }, index=[3, 7, 9, 12])
    original = wide.copy(deep=True)
    masked = mask_ineligible_ocq(wide)
    assert masked.index.equals(wide.index)
    assert masked.loc[[3, 7], ["C08_item01_human", "C08_wording_ai", "C08_preference"]].isna().all().all()
    pd.testing.assert_series_equal(masked.C07_item01_human, wide.C07_item01_human)
    pd.testing.assert_series_equal(masked.C08_chocolate_frequency, wide.C08_chocolate_frequency)
    assert ocq_nonconsumer(wide).tolist() == [True, True, False, False]
    long = pd.DataFrame({"respondent_id": ["citizen:C:3", "citizen:C:3", "citizen:C:9"], "scale_id": ["08", "07", "08"]})
    assert eligible_long_rows(long, wide, "C").index.tolist() == [1, 2]
    pd.testing.assert_frame_equal(wide, original)
