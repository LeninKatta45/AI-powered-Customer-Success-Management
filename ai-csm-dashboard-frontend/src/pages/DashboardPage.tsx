// src/pages/DashboardPage.tsx
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { useEffect, useState, useMemo } from 'react';
import { MoreHorizontal, ArrowUpDown, ChevronLeft, ChevronRight, Sparkles, Wand2 } from "lucide-react"; // Added Wand2 for AI Draft
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { useNavigate } from 'react-router-dom';
import { toast } from "sonner";

// Interfaces
interface Company { id: number; name: string; created_at: string; }
interface UserHealthScore { score: number; status: string; calculated_at: string; }
interface User { id: number; email: string; company_id: number; created_at: string; latest_health_score?: UserHealthScore | null; company_name?: string; }

// Backend API response for manual nudge
interface ManualNudgeResponse {
  message: string;
  ai_generated_draft?: string | null;
  is_draft?: boolean; // Backend will return this
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
const USERS_PER_PAGE = 10;

type SortKey = keyof User | 'health_score' | 'status';
interface SortConfig {
  key: SortKey;
  direction: 'ascending' | 'descending';
}

const DashboardPage = () => {
  const navigate = useNavigate();
  const [allCompanies, setAllCompanies] = useState<Company[]>([]);
  const [allUsers, setAllUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [userSearchTerm, setUserSearchTerm] = useState('');
  const [sortConfig, setSortConfig] = useState<SortConfig | null>({ key: 'id', direction: 'ascending' });
  const [currentPageUsers, setCurrentPageUsers] = useState(1);

  const [isNudgeModalOpen, setIsNudgeModalOpen] = useState(false);
  const [nudgeUser, setNudgeUser] = useState<User | null>(null);
  const [nudgeMessage, setNudgeMessage] = useState(''); // This will be the editable message
  const [nudgeAIAssistTopic, setNudgeAIAssistTopic] = useState('');
  // --- NEW: State for AI Drafting process (Tweak 1) ---
  const [isDraftingWithAI, setIsDraftingWithAI] = useState(false);
  const [isSendingNudge, setIsSendingNudge] = useState(false);


  const [isProcessingAdminAction, setIsProcessingAdminAction] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true); setError(null);
      try {
        const companiesResponse = await fetch(`${API_BASE_URL}/companies/?limit=100`);
        if (!companiesResponse.ok) throw new Error('Failed to fetch companies');
        const companiesData: Company[] = await companiesResponse.json();
        setAllCompanies(companiesData);

        const companyMap = new Map<number, string>();
        companiesData.forEach(company => companyMap.set(company.id, company.name));

        const usersResponse = await fetch(`${API_BASE_URL}/users/?limit=200`);
        if (!usersResponse.ok) throw new Error('Failed to fetch users');
        const usersDataFromApi: Array<User & { company: Company | null }> = await usersResponse.json();
        
        const usersWithDetails = await Promise.all(
          usersDataFromApi.map(async (userFromApi) => {
            const user: User = {
                id: userFromApi.id, email: userFromApi.email, company_id: userFromApi.company_id,
                created_at: userFromApi.created_at,
                company_name: userFromApi.company?.name || companyMap.get(userFromApi.company_id) || 'Unknown Company',
            };
            try {
                const healthScoreRes = await fetch(`${API_BASE_URL}/users/${user.id}/latest_health_score`);
                const healthScoreData = healthScoreRes.ok ? await healthScoreRes.json() : null;
                return { ...user, latest_health_score: healthScoreData };
            } catch (e) { return { ...user, latest_health_score: null }; }
          })
        );
        setAllUsers(usersWithDetails);
      } catch (err) {
        if (err instanceof Error) setError(err.message); else setError('An unknown error occurred');
      } finally { setLoading(false); }
    };
    fetchData();
  }, []);

  const getStatusColorClasses = (status?: string): string => {
    switch (status) {
      case 'At Risk': return 'bg-red-100 text-red-700 border-red-300';
      case 'Power User': return 'bg-green-100 text-green-700 border-green-300';
      case 'Healthy': return 'bg-blue-100 text-blue-700 border-blue-300';
      case 'Neutral': return 'bg-gray-100 text-gray-700 border-gray-300';
      default: return 'bg-slate-100 text-slate-700 border-slate-300';
    }
  };

  const requestSort = (key: SortKey) => {
    let direction: 'ascending' | 'descending' = 'ascending';
    if (sortConfig?.key === key && sortConfig.direction === 'ascending') direction = 'descending';
    setSortConfig({ key, direction }); setCurrentPageUsers(1);
  };

  const sortedAndFilteredUsers = useMemo(() => {
    let items = [...allUsers];
    if (userSearchTerm) {
      const filter = userSearchTerm.toLowerCase();
      items = items.filter(u => u.email.toLowerCase().includes(filter) || u.company_name?.toLowerCase().includes(filter) || u.id.toString().includes(filter));
    }
    if (sortConfig) {
      items.sort((a, b) => {
        let valA: any = ''; let valB: any = '';
        if (sortConfig.key === 'health_score') { valA = a.latest_health_score?.score ?? -Infinity; valB = b.latest_health_score?.score ?? -Infinity; }
        else if (sortConfig.key === 'status') { valA = a.latest_health_score?.status ?? ''; valB = b.latest_health_score?.status ?? ''; }
        else if (sortConfig.key === 'company_name') { valA = a.company_name ?? ''; valB = b.company_name ?? ''; }
        else { valA = (a as any)[sortConfig.key]; valB = (b as any)[sortConfig.key]; }
        valA = (valA === null || valA === undefined) ? (typeof valA === 'number' ? -Infinity : '') : valA;
        valB = (valB === null || valB === undefined) ? (typeof valB === 'number' ? -Infinity : '') : valB;
        if (typeof valA === 'string' && typeof valB === 'string') { valA = valA.toLowerCase(); valB = valB.toLowerCase(); }
        if (valA < valB) return sortConfig.direction === 'ascending' ? -1 : 1;
        if (valA > valB) return sortConfig.direction === 'ascending' ? 1 : -1;
        return 0;
      });
    }
    return items;
  }, [allUsers, userSearchTerm, sortConfig]);

  const totalUserPages = Math.ceil(sortedAndFilteredUsers.length / USERS_PER_PAGE);
  const paginatedUsers = useMemo(() => sortedAndFilteredUsers.slice((currentPageUsers - 1) * USERS_PER_PAGE, currentPageUsers * USERS_PER_PAGE), [sortedAndFilteredUsers, currentPageUsers]);

  const handleOpenNudgeModal = (user: User) => {
    setNudgeUser(user); setNudgeMessage(''); setNudgeAIAssistTopic(''); setIsNudgeModalOpen(true);
  };

  // --- NEW: Handler for AI Draft (Tweak 1) ---
  const handleDraftWithAI = async () => {
    if (!nudgeUser || !nudgeAIAssistTopic.trim()) {
      toast.error("AI Assist Topic Missing", { description: "Please provide a topic for the AI to draft a message."});
      return;
    }
    setIsDraftingWithAI(true);
    setNudgeMessage(''); // Clear any manually typed message if drafting with AI
    toast.info("Drafting with AI...", { description: `Topic: ${nudgeAIAssistTopic}`});
    try {
      const payload = {
        message: "", // Explicitly empty when drafting
        ai_assist_topic: nudgeAIAssistTopic.trim(),
        draft_only: true, // Key flag for backend
      };
      const response = await fetch(`${API_BASE_URL}/users/${nudgeUser.id}/send_manual_nudge`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
      const result: ManualNudgeResponse = await response.json();
      if (!response.ok || !result.is_draft) { // Check for is_draft flag
        throw new Error(result.message || "Failed to generate AI draft.");
      }
      if (result.ai_generated_draft) {
        setNudgeMessage(result.ai_generated_draft); // Populate textarea with AI draft
        toast.success("AI Draft Generated!", { description: "Review and edit the message below before sending."});
      } else {
        toast.warning("AI Draft", { description: "AI returned an empty draft. Please try a different topic or write manually."});
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Unknown error drafting with AI.";
      toast.error("AI Draft Failed", { description: errorMsg });
    } finally {
      setIsDraftingWithAI(false);
    }
  };
  
  // --- MODIFIED: Handler for Sending Final Nudge (Tweak 1) ---
  const handleSendNudge = async () => {
    if (!nudgeUser || !nudgeMessage.trim()) { // Now only checks nudgeMessage
        toast.error("Message Empty", { description: "Please ensure the message content is not empty."});
        return;
    }
    setIsSendingNudge(true);
    toast.info("Sending Nudge...", { description: `To: ${nudgeUser.email}`});
    try {
      // When sending, draft_only is false (or omitted), and message is the final content
      const payload = {
        message: nudgeMessage.trim(),
        ai_assist_topic: nudgeAIAssistTopic.trim() || null, // Can still send topic for logging/context if it was used for draft
        draft_only: false, // Explicitly false or omit
      };
      const response = await fetch(`${API_BASE_URL}/users/${nudgeUser.id}/send_manual_nudge`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
      const result: ManualNudgeResponse = await response.json();
      if (!response.ok || result.is_draft) { // Should not be a draft at this stage
        throw new Error(result.message || `Failed to send nudge. Status: ${response.status}`);
      }
      toast.success(result.message || `Nudge queued for ${nudgeUser.email}!`);
      // Reset modal state and close
      setIsNudgeModalOpen(false); setNudgeUser(null); setNudgeMessage(''); setNudgeAIAssistTopic('');
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Unknown error sending nudge.";
      toast.error("Error Sending Nudge", { description: errorMsg });
    } finally {
      setIsSendingNudge(false);
    }
  };

  const handleAdminAction = async (action: 'healthScores' | 'triggers') => {
    if (isProcessingAdminAction) return;
    setIsProcessingAdminAction(true);
    const endpoint = action === 'healthScores' ? `${API_BASE_URL}/admin/run_daily_health_score_calculations` : `${API_BASE_URL}/admin/run_daily_trigger_processing`;
    const actionName = action === 'healthScores' ? "Calc Health Scores" : "Run Daily Triggers";
    toast.info(`Requesting to ${actionName}...`);
    try {
      const response = await fetch(endpoint, { method: 'POST' });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || `Failed to ${actionName.toLowerCase()}`);
      toast.success(result.message || `${actionName} initiated.`);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : `Unknown error for ${actionName.toLowerCase()}.`;
      toast.error(`Error: ${actionName}`, { description: errorMsg });
    } finally { setIsProcessingAdminAction(false); }
  };

  if (loading) return <div className="flex justify-center items-center min-h-[calc(100vh-10rem)]"><p>Loading dashboard data...</p></div>;
  if (error) return <div className="text-red-600 p-4 bg-red-50 border border-red-200 rounded-md text-center">Error: {error}</div>;

  const SortableTableHead = ({ sortKey, children }: { sortKey: SortKey; children: React.ReactNode }) => (
    <TableHead onClick={() => requestSort(sortKey)} className="cursor-pointer hover:bg-muted/80 transition-colors">
      <div className="flex items-center gap-2">{children}{sortConfig?.key === sortKey ? (<ArrowUpDown className={`h-4 w-4 ${sortConfig.direction === 'descending' ? 'rotate-180' : ''}`} />) : (<ArrowUpDown className="h-4 w-4 opacity-50" />)}</div>
    </TableHead>
  );

  return (
    <div className="space-y-6 lg:space-y-8">
      <header className="flex flex-col sm:flex-row justify-between items-center gap-4">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground">CSM Dashboard</h1>
        <div className="flex space-x-2">
          <Button onClick={() => handleAdminAction('healthScores')} disabled={isProcessingAdminAction} variant="outline" size="sm"><Sparkles className="mr-2 h-4 w-4" /> {isProcessingAdminAction ? "Processing..." : "Calc Health Scores"}</Button>
          <Button onClick={() => handleAdminAction('triggers')} disabled={isProcessingAdminAction} variant="outline" size="sm"><Sparkles className="mr-2 h-4 w-4" /> {isProcessingAdminAction ? "Processing..." : "Run Daily Triggers"}</Button>
        </div>
      </header>

      <section className="grid gap-4 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        <Card><CardHeader className="pb-2"><CardTitle className="text-sm font-medium">Total Companies</CardTitle></CardHeader><CardContent><div className="text-2xl font-bold">{allCompanies.length}</div></CardContent></Card>
        <Card><CardHeader className="pb-2"><CardTitle className="text-sm font-medium">Total Users</CardTitle></CardHeader><CardContent><div className="text-2xl font-bold">{allUsers.length}</div></CardContent></Card>
        <Card><CardHeader className="pb-2"><CardTitle className="text-sm font-medium">At Risk Users</CardTitle></CardHeader><CardContent><div className="text-2xl font-bold text-red-600">{allUsers.filter(u => u.latest_health_score?.status === 'At Risk').length}</div></CardContent></Card>
        <Card><CardHeader className="pb-2"><CardTitle className="text-sm font-medium">Power Users</CardTitle></CardHeader><CardContent><div className="text-2xl font-bold text-green-600">{allUsers.filter(u => u.latest_health_score?.status === 'Power User').length}</div></CardContent></Card>
      </section>

      <section>
        <div className="flex flex-col sm:flex-row items-center justify-between mb-4 gap-3">
          <h2 className="text-xl sm:text-2xl font-semibold">Users Overview</h2>
          <Input type="search" placeholder="Filter users by ID, email, or company..." value={userSearchTerm} onChange={(e) => {setUserSearchTerm(e.target.value); setCurrentPageUsers(1);}} className="max-w-xs sm:max-w-sm w-full sm:w-auto"/>
        </div>
        {paginatedUsers.length > 0 ? (
          <Card className="overflow-hidden">
            <CardContent className="p-0"><div className="overflow-x-auto"><Table>
              <TableHeader><TableRow>
                <SortableTableHead sortKey="id">User ID</SortableTableHead><SortableTableHead sortKey="email">Email</SortableTableHead>
                <SortableTableHead sortKey="company_name">Company</SortableTableHead><SortableTableHead sortKey="health_score">Health Score</SortableTableHead>
                <SortableTableHead sortKey="status">Status</SortableTableHead><TableHead className="text-right">Actions</TableHead>
              </TableRow></TableHeader>
              <TableBody>{paginatedUsers.map((user) => (
                <TableRow key={user.id}>
                  <TableCell className="font-medium">{user.id}</TableCell><TableCell>{user.email}</TableCell>
                  <TableCell>{user.company_name}</TableCell>
                  <TableCell className="text-center">{user.latest_health_score?.score ?? 'N/A'}</TableCell>
                  <TableCell className="text-center">{user.latest_health_score ? (<Badge className={getStatusColorClasses(user.latest_health_score.status)} variant="outline">{user.latest_health_score.status}</Badge>) : (<Badge className={getStatusColorClasses(undefined)} variant="outline">N/A</Badge>)}</TableCell>
                  <TableCell className="text-right">
                    <DropdownMenu><DropdownMenuTrigger asChild><Button variant="ghost" size="icon" className="h-8 w-8"><MoreHorizontal className="h-4 w-4" /></Button></DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuLabel>Actions</DropdownMenuLabel>
                        <DropdownMenuItem onClick={() => navigate(`/users/${user.id}`)}>View User Details</DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem onClick={() => handleOpenNudgeModal(user)}>Send Manual Nudge</DropdownMenuItem>
                      </DropdownMenuContent></DropdownMenu>
                  </TableCell></TableRow>))}
              </TableBody></Table></div></CardContent>
            {totalUserPages > 1 && (
              <div className="flex items-center justify-end space-x-2 py-3 px-4 border-t">
                <Button variant="outline" size="sm" onClick={() => setCurrentPageUsers(p => Math.max(1, p - 1))} disabled={currentPageUsers === 1}><ChevronLeft className="h-4 w-4 mr-1" /> Previous</Button>
                <span className="text-sm text-muted-foreground">Page {currentPageUsers} of {totalUserPages}</span>
                <Button variant="outline" size="sm" onClick={() => setCurrentPageUsers(p => Math.min(totalUserPages, p + 1))} disabled={currentPageUsers === totalUserPages}>Next <ChevronRight className="h-4 w-4 ml-1" /></Button>
              </div>)}
          </Card>
        ) : (<Card className="text-center text-muted-foreground py-8"><CardContent>{userSearchTerm ? "No users match filter." : "No users found."}</CardContent></Card>)}
      </section>

      <section>
        <h2 className="text-xl sm:text-2xl font-semibold mb-4">Companies Overview</h2>
        {allCompanies.length > 0 ? (
          <Card className="overflow-hidden"><CardContent className="p-0"><div className="overflow-x-auto"><Table>
            <TableHeader><TableRow>
              <TableHead className="w-[120px]">Company ID</TableHead><TableHead>Name</TableHead>
              <TableHead>Users</TableHead><TableHead>Created At</TableHead><TableHead className="text-right">Actions</TableHead>
            </TableRow></TableHeader>
            <TableBody>{allCompanies.map(c => (<TableRow key={c.id}>
              <TableCell className="font-medium">{c.id}</TableCell><TableCell>{c.name}</TableCell>
              <TableCell>{allUsers.filter(u => u.company_id === c.id).length}</TableCell>
              <TableCell>{new Date(c.created_at).toLocaleDateString()}</TableCell>
              <TableCell className="text-right">
                <DropdownMenu><DropdownMenuTrigger asChild><Button variant="ghost" size="icon" className="h-8 w-8"><MoreHorizontal className="h-4 w-4" /></Button></DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuLabel>Actions</DropdownMenuLabel>
                    <DropdownMenuItem onClick={() => navigate(`/companies/${c.id}`)}>View Company Details</DropdownMenuItem>
                  </DropdownMenuContent></DropdownMenu>
              </TableCell></TableRow>))}
            </TableBody></Table></div></CardContent></Card>
        ) : (<Card className="text-center text-muted-foreground py-8"><CardContent>No companies found.</CardContent></Card>)}
      </section>

      {/* Manual Nudge Modal - Updated for Two-Step AI Assist */}
      {nudgeUser && (
        <Dialog open={isNudgeModalOpen} onOpenChange={(isOpen) => {
          if ((isDraftingWithAI || isSendingNudge) && !isOpen) return; // Prevent closing while processing
          setIsNudgeModalOpen(isOpen);
          if (!isOpen) { setNudgeUser(null); setNudgeMessage(''); setNudgeAIAssistTopic('');}
        }}>
          <DialogContent className="sm:max-w-lg"> {/* Increased width slightly */}
            <DialogHeader>
              <DialogTitle>Send Manual Nudge to {nudgeUser.email}</DialogTitle>
              <DialogDescription>
                Type your message directly, or provide a topic and click "Draft with AI" to generate a starting point.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-6 py-4"> {/* Increased gap */}
              <div className="space-y-2">
                <Label htmlFor="nudge-ai-topic">AI Assist Topic (Optional)</Label>
                <div className="flex items-center space-x-2">
                    <Input 
                      id="nudge-ai-topic" 
                      value={nudgeAIAssistTopic} 
                      onChange={(e) => setNudgeAIAssistTopic(e.target.value)} 
                      placeholder="e.g., Ask about their onboarding experience"
                      disabled={isDraftingWithAI || isSendingNudge}
                    />
                    <Button 
                        type="button" 
                        variant="outline" 
                        size="sm"
                        onClick={handleDraftWithAI}
                        disabled={!nudgeAIAssistTopic.trim() || isDraftingWithAI || isSendingNudge || !!nudgeMessage.trim()} // Disable if message has content
                    >
                        {isDraftingWithAI ? "Drafting..." : <><Wand2 className="mr-2 h-4 w-4" /> Draft with AI</>}
                    </Button>
                </div>
                {!!nudgeMessage.trim() && (
                    <p className="text-xs text-muted-foreground">Clear message below to enable AI drafting with a new topic.</p>
                 )}
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="nudge-message">Message</Label>
                <Textarea 
                  id="nudge-message" 
                  value={nudgeMessage} 
                  onChange={(e) => setNudgeMessage(e.target.value)} 
                  className="h-40" // Increased height
                  placeholder="Type your message here, or let AI draft it first..."
                  disabled={isDraftingWithAI || isSendingNudge}
                />
              </div>
            </div>
            <DialogFooter className="gap-2 sm:justify-end">
              <Button type="button" variant="outline" onClick={() => setIsNudgeModalOpen(false)} disabled={isDraftingWithAI || isSendingNudge}>Cancel</Button>
              <Button 
                type="button" // Changed from submit to prevent form submission if wrapped in form
                onClick={handleSendNudge} 
                disabled={isDraftingWithAI || isSendingNudge || !nudgeMessage.trim()}
              >
                {isSendingNudge ? "Sending..." : "Send Message"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
};

export default DashboardPage;