"""HCU GLM page leases for asynchronous layer-group KV transfers."""

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator


class HcuLayerPipelineAllocator(PagedTokenToKVPoolAllocator):
    """Keep source pages alive until every Mooncake layer send has completed.

    Pinning and completion draining are scheduler-stream owned. All frees,
    including main's fixed-shape segment/group frees, pass through
    ``_release_page_ids``; the ordinary staged free list stays authoritative.
    """

    def clear(self):
        if getattr(self, "_active_page_leases", 0):
            raise RuntimeError("Cannot clear the KV allocator while pages are pinned")
        super().clear()
        self._active_page_leases = 0
        self._page_pin_counts = torch.zeros(
            self.num_pages + 1, dtype=torch.int32, device=self.device
        )
        self._deferred_free_pages = torch.zeros(
            self.num_pages + 1, dtype=torch.bool, device=self.device
        )

    def _normalize_page_indices(self, page_indices):
        pages = torch.as_tensor(
            page_indices, dtype=torch.int64, device=self.device
        ).reshape(-1)
        pages = torch.unique(pages[pages > 0])
        if self.debug_mode:
            assert torch.all(pages <= self.num_pages)
        return pages

    def pin_pages(self, page_indices):
        pages = self._normalize_page_indices(page_indices)
        if self.debug_mode:
            assert not torch.any(torch.isin(pages, self.get_all_free_pages()))
        self._page_pin_counts[pages] += 1
        self._active_page_leases += pages.numel()
        return pages

    def unpin_pages(self, page_indices):
        pages = self._normalize_page_indices(page_indices)
        counts = self._page_pin_counts[pages]
        if self.debug_mode:
            assert torch.all(counts > 0), "Cannot unpin an unpinned page"
        ready = pages[(counts == 1) & self._deferred_free_pages[pages]]
        self._page_pin_counts[pages] = counts - 1
        self._active_page_leases -= pages.numel()
        self._deferred_free_pages[ready] = False
        if ready.numel():
            super()._release_page_ids(ready)

    def _release_page_ids(self, *page_ids):
        if not self._active_page_leases:
            return super()._release_page_ids(*page_ids)
        pages = self._normalize_page_indices(torch.cat(page_ids))
        pages = pages[~self._deferred_free_pages[pages]]
        pinned = self._page_pin_counts[pages] > 0
        self._deferred_free_pages[pages[pinned]] = True
        ready = pages[~pinned]
        if ready.numel():
            super()._release_page_ids(ready)
