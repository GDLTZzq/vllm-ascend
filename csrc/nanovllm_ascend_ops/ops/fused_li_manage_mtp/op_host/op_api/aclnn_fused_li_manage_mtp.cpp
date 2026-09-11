#include "aclnn_fused_li_manage_mtp.h"
#ifdef __cplusplus
extern "C" {
#endif
extern aclnnStatus aclnnInnerNanovllmFusedLiManageMtpGetWorkspaceSize(
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *,
    uint64_t *, aclOpExecutor **);
extern aclnnStatus aclnnInnerNanovllmFusedLiManageMtp(
    void *, uint64_t, aclOpExecutor *, const aclrtStream);

aclnnStatus aclnnNanovllmFusedLiManageMtpGetWorkspaceSize(
    const aclTensor *indexWeights, const aclTensor *queryDequantScale,
    const aclTensor *query, const aclTensor *indexKeyDequantScale,
    const aclTensor *indexKeyCache, const aclTensor *indexBlockTable,
    const aclTensor *actualSeqLengthsQuery, const aclTensor *actualSeqLengthsKey,
    const aclTensor *offloadSeqLengthsKey, const aclTensor *numCacheTokens,
    const aclTensor *requestState, const aclTensor *reqPoolEntries,
    const aclTensor *cacheSlotsPool, const aclTensor *topkSrcIds,
    const aclTensor *topkDstSlots, const aclTensor *topkMissCounts,
    const aclTensor *missSrcIds, const aclTensor *missDstSlots,
    const aclTensor *missCounts, const aclTensor *cacheSlotsPoolOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerNanovllmFusedLiManageMtpGetWorkspaceSize(
        indexWeights, queryDequantScale, query, indexKeyDequantScale,
        indexKeyCache, indexBlockTable, actualSeqLengthsQuery,
        actualSeqLengthsKey, offloadSeqLengthsKey, numCacheTokens,
        requestState, reqPoolEntries, cacheSlotsPool, topkSrcIds,
        topkDstSlots, topkMissCounts, missSrcIds, missDstSlots, missCounts,
        cacheSlotsPoolOut, workspaceSize, executor);
}
aclnnStatus aclnnNanovllmFusedLiManageMtp(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    const aclrtStream stream)
{
    return aclnnInnerNanovllmFusedLiManageMtp(workspace, workspaceSize, executor, stream);
}
#ifdef __cplusplus
}
#endif
