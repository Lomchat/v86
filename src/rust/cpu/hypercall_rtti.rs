//! MSVC C++ RTTI — __RTDynamicCast (dynamic_cast runtime).
//! Implemented from the publicly documented MSVC RTTI metadata layout and standard dynamic_cast semantics; mirrors src/worker/modules/crt-rtti.ts.

use crate::cpu::cpu::{safe_read32s, safe_read8};

const BCD_NOTVISIBLE: u32 = 0x0000_0002;
const BCD_AMBIGUOUS: u32 = 0x0000_0004;
const CHD_MULTINH: u32 = 0x0000_0001;
const CHD_VIRTINH: u32 = 0x0000_0002;

#[derive(Copy, Clone)]
struct BcdView {
    p_type_descriptor: u32,
    num_contained_bases: u32,
    mdisp: i32,
    pdisp: i32,
    vdisp: i32,
    attributes: u32,
}

struct HierarchyView {
    base_class_array_ptr: u32,
    n_bases: u32,
}

pub enum RtDynamicCastResult {
    Success(u32),
    FailNull,
    FailBadCast,
}

#[inline]
unsafe fn rd32(addr: u32) -> Option<u32> {
    match safe_read32s(addr as i32) {
        Ok(v) => Some(v as u32),
        Err(_) => None,
    }
}

#[inline]
unsafe fn rd8(addr: u32) -> Option<u8> {
    match safe_read8(addr as i32) {
        Ok(v) => Some(v as u8),
        Err(_) => None,
    }
}

unsafe fn types_equal(lhs_ptr: u32, rhs_ptr: u32) -> bool {
    if lhs_ptr == 0 || rhs_ptr == 0 {
        return false;
    }
    if lhs_ptr == rhs_ptr {
        return true;
    }
    let mut saw_non_nul = false;
    for i in 0..256u32 {
        let c1 = match rd8(lhs_ptr + 8 + i) {
            Some(v) => v,
            None => return false,
        };
        let c2 = match rd8(rhs_ptr + 8 + i) {
            Some(v) => v,
            None => return false,
        };
        if c1 != c2 {
            return false;
        }
        if c1 == 0 {
            return saw_non_nul;
        }
        saw_non_nul = true;
    }
    false
}

unsafe fn read_bcd(ptr: u32) -> Option<BcdView> {
    if ptr < 0x1000 {
        return None;
    }
    Some(BcdView {
        p_type_descriptor: rd32(ptr)?,
        num_contained_bases: rd32(ptr + 4)?,
        mdisp: rd32(ptr + 8).map(|v| v as i32)?,
        pdisp: rd32(ptr + 12).map(|v| v as i32)?,
        vdisp: rd32(ptr + 16).map(|v| v as i32)?,
        attributes: rd32(ptr + 20)?,
    })
}

unsafe fn read_bcd_at_index(base_class_array_ptr: u32, index: u32) -> Option<BcdView> {
    if base_class_array_ptr < 0x1000 {
        return None;
    }
    let bcd_ptr = rd32(base_class_array_ptr + index * 4)?;
    read_bcd(bcd_ptr)
}

unsafe fn get_complete_object_locator(inptr: u32) -> Option<u32> {
    if inptr < 4 {
        return None;
    }
    let vfptr = rd32(inptr)?;
    if vfptr < 8 {
        return None;
    }
    rd32(vfptr - 4)
}

unsafe fn find_complete_object(inptr: u32) -> Option<u32> {
    let locator_ptr = get_complete_object_locator(inptr)?;
    let offset = rd32(locator_ptr + 4)?;
    let cd_offset = rd32(locator_ptr + 8)?;
    let mut complete_object = inptr.wrapping_sub(offset);

    let offset_value = if cd_offset > 0 && inptr >= cd_offset {
        rd32(inptr - cd_offset).map(|v| v as i32).unwrap_or(0)
    } else {
        0
    };
    complete_object = complete_object.wrapping_sub(offset_value as u32);
    Some(complete_object)
}

unsafe fn pmd_to_offset(p_this: u32, pmd: &BcdView) -> i32 {
    let mut ret_off = 0i32;
    if pmd.pdisp >= 0 {
        ret_off = pmd.pdisp;
        let vb_ptr_addr = p_this.wrapping_add(ret_off as u32);
        let vb_table = match rd32(vb_ptr_addr) {
            Some(v) => v,
            None => return 0,
        };
        let slot_addr = vb_table.wrapping_add(pmd.vdisp as u32);
        ret_off += match rd32(slot_addr).map(|v| v as i32) {
            Some(v) => v,
            None => return 0,
        };
    }
    ret_off + pmd.mdisp
}

unsafe fn read_hierarchy(locator_ptr: u32) -> Option<HierarchyView> {
    let class_desc_ptr = rd32(locator_ptr + 16)?;
    if class_desc_ptr < 0x1000 {
        return None;
    }
    let base_class_array_ptr = rd32(class_desc_ptr + 12)?;
    let n_bases = rd32(class_desc_ptr + 8)?;
    if base_class_array_ptr == 0 || n_bases == 0 || n_bases > 0xffff {
        return None;
    }
    Some(HierarchyView {
        base_class_array_ptr,
        n_bases,
    })
}

unsafe fn find_cross_cast_target(hier: &HierarchyView, target_type_ptr: u32) -> Option<BcdView> {
    for i in 0..hier.n_bases {
        let bcd = read_bcd_at_index(hier.base_class_array_ptr, i)?;
        if types_equal(bcd.p_type_descriptor, target_type_ptr)
            && (bcd.attributes & (BCD_NOTVISIBLE | BCD_AMBIGUOUS)) == 0
        {
            return Some(bcd);
        }
    }
    None
}

unsafe fn find_si_target_type_instance(
    locator_ptr: u32,
    target_type_ptr: u32,
) -> Option<BcdView> {
    let hier = read_hierarchy(locator_ptr)?;
    for i in 0..hier.n_bases {
        let bcd = read_bcd_at_index(hier.base_class_array_ptr, i)?;
        if types_equal(bcd.p_type_descriptor, target_type_ptr)
            && (bcd.attributes & BCD_NOTVISIBLE) == 0
        {
            return Some(bcd);
        }
    }
    None
}

unsafe fn find_mi_target_type_instance(
    complete_object: u32,
    locator_ptr: u32,
    src_type_ptr: u32,
    src_offset: i32,
    target_type_ptr: u32,
) -> Option<BcdView> {
    let hier = read_hierarchy(locator_ptr)?;

    for i in 0..hier.n_bases {
        let bcd = read_bcd_at_index(hier.base_class_array_ptr, i)?;
        if !types_equal(bcd.p_type_descriptor, target_type_ptr) {
            continue;
        }

        let n_contained = bcd.num_contained_bases.min(hier.n_bases - i - 1);
        for j in 0..n_contained {
            let sub = read_bcd_at_index(hier.base_class_array_ptr, i + 1 + j)?;
            if types_equal(sub.p_type_descriptor, src_type_ptr)
                && pmd_to_offset(complete_object, &sub) == src_offset
            {
                return Some(bcd);
            }
        }
    }

    find_cross_cast_target(&hier, target_type_ptr)
}

unsafe fn find_vi_target_type_instance(
    complete_object: u32,
    locator_ptr: u32,
    src_type_ptr: u32,
    src_offset: i32,
    target_type_ptr: u32,
) -> Option<BcdView> {
    let hier = read_hierarchy(locator_ptr)?;
    let mut result: Option<BcdView> = None;

    for i in 0..hier.n_bases {
        let bcd = read_bcd_at_index(hier.base_class_array_ptr, i)?;
        if !types_equal(bcd.p_type_descriptor, target_type_ptr) {
            continue;
        }

        let n_contained = bcd.num_contained_bases.min(hier.n_bases - i - 1);
        for j in 0..n_contained {
            let sub = read_bcd_at_index(hier.base_class_array_ptr, i + 1 + j)?;
            if types_equal(sub.p_type_descriptor, src_type_ptr)
                && pmd_to_offset(complete_object, &sub) == src_offset
            {
                if let Some(prev) = result {
                    if pmd_to_offset(complete_object, &prev) != pmd_to_offset(complete_object, &bcd)
                    {
                        return None;
                    }
                }
                result = Some(bcd);
            }
        }
    }

    if let Some(bcd) = result {
        return Some(bcd);
    }
    find_cross_cast_target(&hier, target_type_ptr)
}

pub unsafe fn rt_dynamic_cast(
    inptr: u32,
    vf_delta: i32,
    src_type_ptr: u32,
    target_type_ptr: u32,
    is_reference: i32,
) -> RtDynamicCastResult {
    if inptr == 0 {
        return RtDynamicCastResult::FailNull;
    }

    let complete_object = match find_complete_object(inptr) {
        Some(v) => v,
        None => return fail_cast(is_reference, inptr),
    };
    let locator_ptr = match get_complete_object_locator(inptr) {
        Some(v) => v,
        None => return fail_cast(is_reference, inptr),
    };

    let class_desc_ptr = match rd32(locator_ptr + 16) {
        Some(v) => v,
        None => return fail_cast(is_reference, inptr),
    };
    if class_desc_ptr < 0x1000 {
        return fail_cast(is_reference, inptr);
    }
    let chd_attributes = match rd32(class_desc_ptr + 4) {
        Some(v) => v,
        None => return fail_cast(is_reference, inptr),
    };

    let adjusted_inptr = inptr.wrapping_sub(vf_delta as u32);
    let inptr_delta = adjusted_inptr.wrapping_sub(complete_object) as i32;

    let base_class = if (chd_attributes & CHD_MULTINH) == 0 {
        find_si_target_type_instance(locator_ptr, target_type_ptr)
    } else if (chd_attributes & CHD_VIRTINH) == 0 {
        find_mi_target_type_instance(
            complete_object,
            locator_ptr,
            src_type_ptr,
            inptr_delta,
            target_type_ptr,
        )
    } else {
        find_vi_target_type_instance(
            complete_object,
            locator_ptr,
            src_type_ptr,
            inptr_delta,
            target_type_ptr,
        )
    };

    match base_class {
        Some(bcd) => {
            let offset = pmd_to_offset(complete_object, &bcd);
            RtDynamicCastResult::Success(complete_object.wrapping_add(offset as u32))
        }
        None => fail_cast(is_reference, inptr),
    }
}

#[inline]
fn fail_cast(is_reference: i32, inptr: u32) -> RtDynamicCastResult {
    if is_reference != 0 && inptr != 0 {
        RtDynamicCastResult::FailBadCast
    } else {
        RtDynamicCastResult::FailNull
    }
}
